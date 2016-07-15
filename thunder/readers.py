import errno
import fnmatch
import glob
import os
import logging

# library reorganization between Python 2 and 3
try:
    from urllib.parse import urlparse
    from urllib.request import url2pathname
except ImportError:
    from urlparse import urlparse
    from urllib import url2pathname
from six.moves import filter
from six import next

from .utils import check_spark
spark = check_spark()
logging.getLogger('boto').setLevel(logging.CRITICAL)

def addextension(path, ext=None):
    """
    Helper function for handling of paths given separately passed file extensions.
    """
    if ext:
        if '*' in path:
            return path
        elif os.path.splitext(path)[1]:
            return path
        else:
            if not ext.startswith('.'):
                ext = '.'+ext
            if not path.endswith(ext):
                if not path.endswith(os.path.sep):
                    path += os.path.sep
                return path + '*' + ext
            else:
                return path
    else:
        return path

def select(files, start, stop):
    """
    Helper function for handling start and stop indices
    """
    if start or stop:
        if start is None:
            start = 0
        if stop is None:
            stop = len(files)
        files = files[start:stop]
    return files

def readlocal(path, offset=None, size=-1):
    """
    Wrapper around open(path, 'rb') that returns the contents of the file as a string.

    Will rethrow FileNotFoundError if it receives an IOError.
    """
    try:
        with open(path, 'rb') as f:
            if offset:
                f.seek(offset)
            buf = f.read(size)
    except IOError as e:
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(e)
        else:
            raise
    return buf

def listrecursive(path, ext=None):
    """
    List files recurisvely
    """
    filenames = set()
    for root, dirs, files in os.walk(path):
        if ext:
            files = fnmatch.filter(files, '*.' + ext)
        for filename in files:
            filenames.add(os.path.join(root, filename))
    filenames = list(filenames)
    filenames.sort()
    return sorted(filenames)

def listflat(path, ext=None):
    """
    List files without recursion
    """
    if os.path.isdir(path):
        if ext:
            files = glob.glob(os.path.join(path, '*.' + ext))
        else:
            files = [os.path.join(path, fname) for fname in os.listdir(path)]
    else:
        files = glob.glob(path)
    # filter out directories
    files = [fpath for fpath in files if not os.path.isdir(fpath)]
    return sorted(files)

def uri_to_path(uri):
    path = url2pathname(urlparse(uri).path)
    if uri and (not path):
        raise ValueError("Could not interpret %s as URI. " +
                         "Paths in URIs should start with 'file:///', not 'file://'")
    return path


class FileNotFoundError(IOError):
    """
    An exception to be thrown when reader implementations can't find a requested file.
    """
    pass


class LocalParallelReader(object):
    """
    Parallel reader backed by python's native file() objects.
    """
    def __init__(self, engine=None, **kwargs):
        self.engine = engine
        self.nfiles = None

    @staticmethod
    def list(path, ext=None, start=None, stop=None, recursive=False):
        """
        Get sorted list of file paths matching path and extension
        """
        files = listflat(path, ext) if not recursive else listrecursive(path, ext)
        if len(files) < 1:
            raise FileNotFoundError('Cannot find files of type "%s" in %s'
                                    % (ext if ext else '*', path))
        files = select(files, start, stop)

        return files

    def read(self, path, ext=None, start=None, stop=None, recursive=False, npartitions=None):
        """
        Sets up Spark RDD across files specified by dataPath on local filesystem.

        Returns RDD of <integer file index, string buffer> k/v pairs.
        """
        path = uri_to_path(path)
        files = self.list(path, ext=ext, start=start, stop=stop, recursive=recursive)

        nfiles = len(files)
        self.nfiles = nfiles

        if spark and isinstance(self.engine, spark):
            npartitions = min(npartitions, nfiles) if npartitions else nfiles
            rdd = self.engine.parallelize(enumerate(files), npartitions)
            return rdd.map(lambda kv: (kv[0], readlocal(kv[1]), kv[1]))
        else:
            return [(k, readlocal(v), v) for k, v in enumerate(files)]


class LocalFileReader(object):
    """
    File reader backed by python's native file() objects.
    """
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def list(path, filename=None, start=None, stop=None, recursive=False, directories=False):
        """
        List files specified by dataPath.

        Datapath may include a single wildcard ('*') in the filename specifier.

        Returns sorted list of absolute path strings.
        """
        path = uri_to_path(path)

        if not filename and recursive:
            return listrecursive(path)

        if filename:
            if os.path.isdir(path):
                path = os.path.join(path, filename)
            else:
                path = os.path.join(os.path.dirname(path), filename)
        else:
            if os.path.isdir(path) and not directories:
                path = os.path.join(path, "*")

        files = glob.glob(path)

        if not directories:
            files = [fpath for fpath in files if not os.path.isdir(fpath)]

        files.sort()
        files = select(files, start, stop)
        return files

    def read(self, path, filename=None, offset=None, size=-1):
        filenames = self.list(path, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % path)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % path)

        return readlocal(filenames[0], offset=offset, size=size)

    def open(self, path, filename=None):
        filenames = self.list(path, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % path)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % path)

        return open(filenames[0], 'rb')


class BotoClient(object):
    """
    Superclass for boto-based S3 and Google storage readers.
    """
    def __init__(self, credentials=None):
        self.credentials = credentials if credentials else {'access': None, 'secret': None}

    @staticmethod
    def parse_query(query, delim='/'):
        """
        Parse a boto query
        """
        key = ''
        prefix = ''
        postfix = ''

        parsed = urlparse(query)
        query = parsed.path.lstrip(delim)
        bucket = parsed.netloc

        if not parsed.scheme.lower() in ('', "gs", "s3", "s3n"):
            raise ValueError("Query scheme must be one of '', 'gs', 's3', or 's3n'; "
                             "got: '%s'" % parsed.scheme)
        storage = parsed.scheme.lower()

        if not bucket.strip() and query:
            toks = query.split(delim, 1)
            bucket = toks[0]
            if len(toks) == 2:
                key = toks[1]
            else:
                key = ''

        if not bucket.strip():
            raise ValueError("Could not parse bucket name from query string '%s'" % query)

        tokens = query.split("*")
        n = len(tokens)
        if n == 0:
            pass
        elif n == 1:
            key = tokens[0]
        elif n == 2:
            index = tokens[0].rfind(delim)
            if index >= 0:
                key = tokens[0][:(index + 1)]
                prefix = tokens[0][(index + 1):] if len(tokens[0]) > (index + 1) else ''
            else:
                prefix = tokens[0]
            postfix = tokens[1]
        else:
            raise ValueError("Only one wildcard ('*') allowed in query string, got: '%s'" % query)

        return storage, bucket, key, prefix, postfix

    @staticmethod
    def check_prefix(bucket, keyPath, delim='/'):
        return len(bucket.get_all_keys(prefix=keyPath, delimiter=delim, max_keys=1)) > 0

    @staticmethod
    def filter_predicate(key, post, inclusive=False):
        kname = key.name
        ends_with_postfix = kname.endswith(post)
        return ends_with_postfix if inclusive else not ends_with_postfix

    @staticmethod
    def retrieve_keys(bucket, key, prefix='', postfix='', delim='/',
                      directories=False, recursive=False):
        """
        Retrieve keys from a bucket
        """
        if key and prefix:
            assert key.endswith(delim)

        key += prefix
        # check whether key is a directory
        if not key.endswith(delim) and key:
            # check for matching prefix
            if BotoClient.check_prefix(bucket, key + delim, delim=delim):
                # found a directory
                key += delim

        listdelim = delim if not recursive else None
        results = bucket.list(prefix=key, delimiter=listdelim)
        if postfix:
            func = lambda k_: BotoClient.filter_predicate(k_, postfix, inclusive=True)
            return filter(func, results)
        elif not directories:
            func = lambda k_: BotoClient.filter_predicate(k_, delim, inclusive=False)
            return filter(func, results)
        else:
            return results


class BotoParallelReader(BotoClient):
    """
    Parallel reader backed by boto AWS client library.
    """
    def __init__(self, engine, credentials=None):
        super(BotoParallelReader, self).__init__(credentials=credentials)
        self.engine = engine
        self.nfiles = None

    def getfiles(self, path, ext=None, start=None, stop=None, recursive=False):
        """
        Get scheme, bucket, and keys for a set of files
        """
        from .utils import connection_with_anon, connection_with_gs

        parse = BotoClient.parse_query(path)

        scheme = parse[0]
        bucket_name = parse[1]

        if scheme == 's3' or scheme == 's3n':
            conn = connection_with_anon(self.credentials)
            bucket = conn.get_bucket(parse[1])
        elif scheme == 'gs':
            conn = connection_with_gs(bucket_name)
            bucket = conn.get_bucket()
        else:
            raise NotImplementedError("No file reader implementation for URL scheme " + scheme)

        keys = BotoClient.retrieve_keys(
            bucket, parse[2], prefix=parse[3], postfix=parse[4], recursive=recursive)
        keylist = [key.name for key in keys]
        if ext:
            keylist = [keyname for keyname in keylist if keyname.endswith(ext)]
        keylist.sort()
        keylist = select(keylist, start, stop)

        return scheme, bucket.name, keylist

    def list(self, dataPath, ext=None, start=None, stop=None, recursive=False):
        """
        List files from remote storage
        """
        scheme, bucket_name, keylist = self.getfiles(
            dataPath, ext=ext, start=start, stop=stop, recursive=recursive)

        return ["%s:///%s/%s" % (scheme, bucket_name, key) for key in keylist]

    def read(self, path, ext=None, start=None, stop=None, recursive=False, npartitions=None):
        """
        Sets up Spark RDD across S3 or GS objects specified by dataPath.

        Returns RDD of <string bucket keyname, string buffer> k/v pairs.
        """
        from .utils import connection_with_anon, connection_with_gs

        path = addextension(path, ext)
        scheme, bucket_name, keylist = self.getfiles(
            path, start=start, stop=stop, recursive=recursive)

        if not keylist:
            raise FileNotFoundError("No objects found for '%s'" % path)

        credentials = self.credentials

        self.nfiles = len(keylist)

        if spark and isinstance(self.engine, spark):

            def getsplit(kvIter):
                if scheme == 's3' or scheme == 's3n':
                    conn = connection_with_anon(credentials)
                    bucket = conn.get_bucket(bucket_name)
                elif scheme == 'gs':
                    conn = boto.storage_uri(bucket_name, 'gs')
                    bucket = conn.get_bucket()
                else:
                    raise NotImplementedError("No file reader implementation for URL scheme " + scheme)

                for kv in kvIter:
                    idx, keyname = kv
                    key = bucket.get_key(keyname)
                    buf = key.get_contents_as_string()
                    yield idx, buf, keyname

            npartitions = min(npartitions, self.nfiles) if npartitions else self.nfiles
            rdd = self.engine.parallelize(enumerate(keylist), npartitions)
            return rdd.mapPartitions(getsplit)

        else:

            if scheme == 's3' or scheme == 's3n':
                conn = connection_with_anon(credentials)
                bucket = conn.get_bucket(bucket_name)
            elif scheme == 'gs':
                conn = connection_with_gs(bucket_name)
                bucket = conn.get_bucket()
            else:
                raise NotImplementedError("No file reader implementation for URL scheme " + scheme)

            def getsplit(kv):
                idx, keyName = kv
                key = bucket.get_key(keyName)
                buf = key.get_contents_as_string()
                return idx, buf, keyName

            return [getsplit(kv) for kv in enumerate(keylist)]


class BotoFileReader(BotoClient):
    """
    File reader backed by boto.
    """
    def getkeys(self, path, filename=None, directories=False, recursive=False):
        """
        Get matching keys for a path
        """
        from .utils import connection_with_anon, connection_with_gs

        parse = BotoClient.parse_query(path)

        scheme = parse[0]
        bucket_name = parse[1]
        key = parse[2]

        if scheme == 's3' or scheme == 's3n':
            conn = connection_with_anon(self.credentials)
            bucket = conn.get_bucket(bucket_name)
        elif scheme == 'gs':
            conn = connection_with_gs(bucket_name)
            bucket = conn.get_bucket()
        else:
            raise NotImplementedError("No file reader implementation for URL scheme " + scheme)

        if filename:
            if not key.endswith("/"):
                if self.check_prefix(bucket, key + "/"):
                    key += "/"
                else:
                    index = key.rfind("/")
                    if index >= 0:
                        key = key[:(index+1)]
                    else:
                        key = ""
            key += filename

        keylist = BotoClient.retrieve_keys(bucket, key, prefix=parse[3], postfix=parse[4],
                                           directories=directories, recursive=recursive)
        return scheme, keylist

    def getkey(self, path, filename=None):
        """
        Get single matching key for a path
        """
        scheme, keys = self.getkeys(path, filename=filename)
        try:
            key = next(keys)
        except StopIteration:
            raise FileNotFoundError("Could not find object for: '%s'" % path)

        # we expect to only have a single key returned
        nextKey = None
        try:
            nextKey = next(keys)
        except StopIteration:
            pass
        if nextKey:
            raise ValueError("Found multiple keys for: '%s'" % path)
        return scheme, key

    def list(self, path, filename=None, start=None, stop=None, recursive=False, directories=False):
        """
        List objects specified by path.

        Returns sorted list of 'gs://' or 's3n://' URIs.
        """
        storageScheme, keys = self.getkeys(
            path, filename=filename, directories=directories, recursive=recursive)
        keys = [storageScheme + ":///" + key.bucket.name + "/" + key.name for key in keys]
        keys.sort()
        keys = select(keys, start, stop)
        return keys

    def read(self, path, filename=None, offset=None, size=-1):
        """
        Read a file specified by path.
        """
        storageScheme, key = self.getkey(path, filename=filename)

        if offset or (size > -1):
            if not offset:
                offset = 0
            if size > -1:
                sizeStr = offset + size - 1  # range header is inclusive
            else:
                sizeStr = ""
            headers = {"Range": "bytes=%d-%s" % (offset, sizeStr)}
            return key.get_contents_as_string(headers=headers)
        else:
            return key.get_contents_as_string()

    def open(self, path, filename=None):
        """
        Open a file specified by path.
        """
        scheme, key = self.getkey(path, filename=filename)
        return BotoReadFileHandle(scheme, key)


class BotoReadFileHandle(object):
    """
    Read-only file handle-like object exposing a subset of file methods.

    Returned by BotoFileReader's open() method.
    """
    def __init__(self, scheme, key):
        self._scheme = scheme
        self._key = key
        self._closed = False
        self._offset = 0

    def close(self):
        try:
            self._key.close(fast=True)
        except TypeError:
            # workaround for early versions of boto that don't have the 'fast' keyword
            self._key.close()
        self._closed = True

    def read(self, size=-1):
        if self._offset or (size > -1):
            # return empty string to indicate EOF if we are offset past the end of the file
            # else boto will throw an error at us
            if self._offset >= self._key.size:
                return ""
            if size > -1:
                sizeStr = str(self._offset + size - 1)  # range header is inclusive
            else:
                sizeStr = ""
            hdrs = {"Range": "bytes=%d-%s" % (self._offset, sizeStr)}
        else:
            hdrs = {}
        buf = self._key.get_contents_as_string(headers=hdrs)
        self._offset += len(buf)
        return buf

    def seek(self, offset, whence=0):
        if whence == 0:
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = self._key.size + offset
        else:
            raise IOError("Invalid 'whence' argument, must be 0, 1, or 2. See file().seek.")

    def tell(self):
        return self._offset

    @property
    def closed(self):
        return self._closed

    @property
    def name(self):
        return self._scheme + ":///" + self._key.bucket.name + "/" + self._key.name

    @property
    def mode(self):
        return "rb"


SCHEMAS_TO_PARALLELREADERS = {
    '': LocalParallelReader,
    'file': LocalParallelReader,
    'gs': BotoParallelReader,
    's3': BotoParallelReader,
    's3n': BotoParallelReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_FILEREADERS = {
    '': LocalFileReader,
    'file': LocalFileReader,
    'gs': BotoFileReader,
    's3': BotoFileReader,
    's3n': BotoFileReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

def normalize_scheme(path, ext):
    """
    Normalize scheme for paths related to hdfs
    """
    path = addextension(path, ext)

    parsed = urlparse(path)
    if parsed.scheme:
        # this appears to already be a fully-qualified URI
        return path
    else:
        # this looks like a local path spec
        import os
        dirname, filename = os.path.split(path)
        if not os.path.isabs(dirname):
            # need to make relative local paths absolute
            dirname = os.path.abspath(dirname)
            path = os.path.join(dirname, filename)
        return "file://" + path

def get_by_scheme(path, lookup, default):
    """
    Helper function used by get*ForPath().
    """
    parsed = urlparse(path)
    class_name = lookup.get(parsed.scheme, default)
    if class_name is None:
        raise NotImplementedError("No implementation for scheme " + parsed.scheme)
    return class_name


def get_parallel_reader(path):
    """
    Returns the class of a parallel reader suitable for the scheme in path.

    The resulting class object must still be instantiated.
    Throws NotImplementedError if the requested scheme is not supported (e.g. "ftp://").
    Returns LocalParallelReader if scheme is absent or not recognized.
    """
    return get_by_scheme(path, SCHEMAS_TO_PARALLELREADERS, LocalParallelReader)


def get_file_reader(path):
    """
    Returns the class of a file reader suitable for the scheme in path.

    The resulting class object must still be instantiated.
    Throws NotImplementedError if the requested scheme is not supported (e.g. "ftp://").
    Returns LocalFileReader if scheme is absent or not recognized.
    """
    return get_by_scheme(path, SCHEMAS_TO_FILEREADERS, LocalFileReader)
