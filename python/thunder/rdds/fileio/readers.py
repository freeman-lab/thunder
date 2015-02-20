"""Classes that abstract reading from various types of filesystems.

Currently two types of 'filesystem' are supported:

* the local file system, via python's native file() objects

* Amazon's S3, using the boto library (only if boto is installed; boto is not a requirement)

For each filesystem, two types of reader classes are provided:

* parallel readers are intended to serve as the entry point to a Spark workflow. They provide a read() method
that itself calls the spark context's parallelize() method, setting up a workflow with one partition per file. This
method returns a Spark RDD of <string filename, string binary data>.

* file readers are intended to abstract across the supported filesystems, providing a consistent interface to several
common file and filesystem operations. These include listing files in a directory, reading the contents of a file,
and providing a file handle or handle-like object that itself supports read(), seek(), and tell() operations.

The reader classes also all support a common syntax for path specifications, including both "standard" file paths
and "URI-like" syntax with an explicitly specified scheme (for instance, "file://" or "s3n://"). This path specification
syntax allows a single wildcard "*" character in the filename, making possible paths like
"s3n:///my-bucket/key-one/foo*.bar", referring to "every object in the S3 bucket my-bucket whose key starts with
'key-one/foo' and ends with '.bar'".
"""
import errno
import fnmatch
import glob
import itertools
import os
import urllib
import urlparse

from thunder.utils.common import AWSCredentials

_haveBoto = False
try:
    import boto
    _haveBoto = True
except ImportError:
    boto = None


class FileNotFoundError(IOError):
    """An exception to be thrown when reader implementations can't find a requested file.

    Implementations are responsible for watching for their own appropriate exceptions and rethrowing
    FileNotFoundError.

    See PEP 3151 for background and inspiration.
    """
    pass


def appendExtensionToPathSpec(dataPath, ext=None):
    """Helper function for consistent handling of paths given with separately passed file extensions

    Returns
    -------
    result: string dataPath
        dataPath string formed by concatenating passed `dataPath` with "*" and passed `ext`, with some
        normalization as appropriate
    """
    if ext:
        if '*' in dataPath:
            # we already have a literal wildcard, which we take as a sign that the user knows
            # what they're doing and don't want us overriding their path by appending extensions to it
            return dataPath
        elif os.path.splitext(dataPath)[1]:
            # looks like we already have a literal extension specified at the end of dataPath.
            # go with that.
            return dataPath
        else:
            # no wildcard in path yet
            # check whether we already end in `ext`, which suggests we've been passed a literal filename.
            # prepend '.' to ext, as mild protection against the case where we have a directory 'bin' and
            # are looking in it for files named '*.bin'.
            if not ext.startswith('.'):
                ext = '.'+ext
            if not dataPath.endswith(ext):
                # we have an extension and we'd like to append it.
                # we assume that dataPath should be pointing to a directory at this point, but we might
                # or might not have a directory separator at the end of it. add it if we don't.
                if not dataPath.endswith(os.path.sep):
                    dataPath += os.path.sep
                # return a path with "/*."+`ext` added to it.
                return dataPath+'*'+ext
            else:
                # we are asking to append `ext`, but it looks like dataPath already ends with '.'+`ext`
                return dataPath
    else:
        return dataPath


def selectByStartAndStopIndices(files, startIdx, stopIdx):
    """Helper function for consistent handling of start and stop indices
    """
    if startIdx or stopIdx:
        if startIdx is None:
            startIdx = 0
        if stopIdx is None:
            stopIdx = len(files)
        files = files[startIdx:stopIdx]
    return files


def _localRead(filePath, startOffset=None, size=-1):
    """Wrapper around open(filepath, 'rb') that returns the contents of the file as a string.

    Will rethrow FileNotFoundError if it receives an IOError with error number indicating that the file isn't found.
    """
    try:
        with open(filePath, 'rb') as f:
            if startOffset:
                f.seek(startOffset)
            buf = f.read(size)
    except IOError, e:
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(e)
        else:
            raise
    return buf


class LocalFSParallelReader(object):
    """Parallel reader backed by python's native file() objects.
    """
    def __init__(self, sparkContext, **kwargs):
        # kwargs allow AWS credentials to be passed into generic Readers w/o exceptions being raised
        # in this case kwargs are just ignored
        self.sc = sparkContext
        self.lastNRecs = None

    @staticmethod
    def uriToPath(uri):
        # thanks stack overflow:
        # http://stackoverflow.com/questions/5977576/is-there-a-convenient-way-to-map-a-file-uri-to-os-path
        path = urllib.url2pathname(urlparse.urlparse(uri).path)
        if uri and (not path):
            # passed a nonempty uri, got an empty path back
            # this happens when given a file uri that starts with "file://" instead of "file:///"
            # error here to prevent unexpected behavior of looking at current working directory
            raise ValueError("Could not interpret %s as URI. " +
                             "Note absolute paths in URIs should start with 'file:///', not 'file://'")
        return path

    @staticmethod
    def _listFilesRecursive(absPath, ext=None):
        filenames = set()
        for root, dirs, files in os.walk(absPath):
            if ext:
                files = fnmatch.filter(files, '*.' + ext)
            for filename in files:
                filenames.add(os.path.join(root, filename))
        filenames = list(filenames)
        filenames.sort()
        return sorted(filenames)

    @staticmethod
    def _listFilesNonRecursive(absPath, ext=None):
        if os.path.isdir(absPath):
            if ext:
                files = glob.glob(os.path.join(absPath, '*.' + ext))
            else:
                files = [os.path.join(absPath, fname) for fname in os.listdir(absPath)]
        else:
            files = glob.glob(absPath)
        # filter out directories
        files = [fpath for fpath in files if not os.path.isdir(fpath)]
        return sorted(files)

    def listFiles(self, absPath, ext=None, startIdx=None, stopIdx=None, recursive=False):
        """Get sorted list of file paths matching passed `absPath` path and `ext` filename extension
        """
        files = LocalFSParallelReader._listFilesNonRecursive(absPath, ext) if not recursive else \
            LocalFSParallelReader._listFilesRecursive(absPath, ext)
        if len(files) < 1:
            raise FileNotFoundError('cannot find files of type "%s" in %s' % (ext if ext else '*', absPath))
        files = selectByStartAndStopIndices(files, startIdx, stopIdx)

        return files

    def read(self, dataPath, ext=None, startIdx=None, stopIdx=None, recursive=False, npartitions=None):
        """Sets up Spark RDD across files specified by dataPath on local filesystem.

        Returns RDD of <integer file index, string buffer> k/v pairs.
        """
        absPath = self.uriToPath(dataPath)
        filePaths = self.listFiles(absPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)

        lfilepaths = len(filePaths)
        self.lastNRecs = lfilepaths
        npartitions = min(npartitions, lfilepaths) if npartitions else lfilepaths
        return self.sc.parallelize(enumerate(filePaths), npartitions).map(lambda (k, v): (k, _localRead(v)))


class _BotoS3Client(object):
    """Superclass for boto-based S3 readers.
    """
    @staticmethod
    def parseS3Query(query, delim='/'):
        keyName = ''
        prefix = ''
        postfix = ''

        parseResult = urlparse.urlparse(query)
        bucketName = parseResult.netloc
        keyQuery = parseResult.path.lstrip(delim)

        if not parseResult.scheme.lower() in ('', "s3", "s3n"):
            raise ValueError("Query scheme must be one of '', 's3', or 's3n'; got: '%s'" % parseResult.scheme)

        # special case handling for strings of form "/bucket/dir":
        if (not bucketName.strip()) and keyQuery:
            toks = keyQuery.split(delim, 1)
            bucketName = toks[0]
            if len(toks) == 2:
                keyQuery = toks[1]
            else:
                keyQuery = ''

        if not bucketName.strip():
            raise ValueError("Could not parse bucket name from query string '%s'" % query)

        keyToks = keyQuery.split("*")
        nkeyToks = len(keyToks)
        if nkeyToks == 0:
            pass
        elif nkeyToks == 1:
            keyName = keyToks[0]
        elif nkeyToks == 2:
            rdelimIdx = keyToks[0].rfind(delim)
            if rdelimIdx >= 0:
                keyName = keyToks[0][:(rdelimIdx+1)]
                prefix = keyToks[0][(rdelimIdx+1):] if len(keyToks[0]) > (rdelimIdx+1) else ''
            else:
                prefix = keyToks[0]
            postfix = keyToks[1]
        else:
            raise ValueError("Only one wildcard ('*') allowed in query string, got: '%s'" % query)

        return bucketName, keyName, prefix, postfix

    @staticmethod
    def checkPrefix(bucket, keyPath, delim='/'):
        return len(bucket.get_all_keys(prefix=keyPath, delimiter=delim, max_keys=1)) > 0

    @staticmethod
    def filterPredicate(key, post, inclusive=False):
        kname = key.name
        retval = not inclusive
        if kname.endswith(post):
            retval = not retval

        return retval

    @staticmethod
    def retrieveKeys(bucket, key, prefix='', postfix='', delim='/', excludeDirectories=True):
        if key and prefix:
            assert key.endswith(delim)

        keyPath = key+prefix
        # if we are asking for a key that doesn't end in a delimiter, check whether it might
        # actually be a directory
        if not keyPath.endswith(delim) and keyPath:
            # not all directories have actual keys associated with them
            # check for matching prefix instead of literal key:
            if _BotoS3Client.checkPrefix(bucket, keyPath+delim, delim=delim):
                # found a directory; change path so that it explicitly refers to directory
                keyPath += delim

        results = bucket.list(prefix=keyPath, delimiter=delim)
        if postfix:
            return itertools.ifilter(lambda k_: _BotoS3Client.filterPredicate(k_, postfix, inclusive=True), results)
        elif excludeDirectories:
            return itertools.ifilter(lambda k_: _BotoS3Client.filterPredicate(k_, delim, inclusive=False), results)
        else:
            return results

    def __init__(self, awsCredentialsOverride=None):
        """Initialization; validates that AWS keys are available as environment variables.

        Will let boto library look up credentials itself according to its own rules - e.g. first looking for
        AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, then going through several possible config files and finally
        looking for a ~/.aws/credentials .ini-formatted file. See boto docs:
        http://boto.readthedocs.org/en/latest/boto_config_tut.html

        However, if an AWSCredentials object is provided, its `awsAccessKeyId` and `awsSecretAccessKey` attributes
        will be used instead of those found by the standard boto credential lookup process.
        """
        if not _haveBoto:
            raise ValueError("The boto package does not appear to be available; boto is required for BotoS3Reader")
        self.awsCredentialsOverride = awsCredentialsOverride


class BotoS3ParallelReader(_BotoS3Client):
    """Parallel reader backed by boto AWS client library.
    """
    def __init__(self, sparkContext, awsCredentialsOverride=None):
        super(BotoS3ParallelReader, self).__init__(awsCredentialsOverride=awsCredentialsOverride)
        self.sc = sparkContext
        self.lastNRecs = None

    def listFiles(self, dataPath, ext=None, startIdx=None, stopIdx=None, recursive=False):
        bucketname, keyNames = self._listFilesImpl(dataPath, ext=ext, startIdx=startIdx, stopIdx=stopIdx,
                                                   recursive=recursive)
        return ["s3n:///%s/%s" % (bucketname, keyname) for keyname in keyNames]

    def _listFilesImpl(self, dataPath, ext=None, startIdx=None, stopIdx=None, recursive=False):
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")
        parse = _BotoS3Client.parseS3Query(dataPath)


        conn = boto.connect_s3(**AWSCredentials.getCredentialsAsDict(self.awsCredentialsOverride))
        bucket = conn.get_bucket(parse[0])
        keys = _BotoS3Client.retrieveKeys(bucket, parse[1], prefix=parse[2], postfix=parse[3])
        keyNameList = [key.name for key in keys]
        if ext:
            keyNameList = [keyname for keyname in keyNameList if keyname.endswith(ext)]
        keyNameList.sort()
        keyNameList = selectByStartAndStopIndices(keyNameList, startIdx, stopIdx)

        return bucket.name, keyNameList

    def read(self, dataPath, ext=None, startIdx=None, stopIdx=None, recursive=False, npartitions=None):
        """Sets up Spark RDD across S3 objects specified by dataPath.

        Returns RDD of <string s3 keyname, string buffer> k/v pairs.
        """
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")
        dataPath = appendExtensionToPathSpec(dataPath, ext)
        bucketName, keyNameList = self._listFilesImpl(dataPath, startIdx=startIdx, stopIdx=stopIdx)

        if not keyNameList:
            raise FileNotFoundError("No S3 objects found for '%s'" % dataPath)

        # try to prevent self from getting pulled into the closure
        awsAccessKeyIdOverride_, awsSecretAccessKeyOverride_ = \
            AWSCredentials.getCredentials(self.awsCredentialsOverride)

        def readSplitFromS3(kvIter):
            conn = boto.connect_s3(aws_access_key_id=awsAccessKeyIdOverride_,
                                   aws_secret_access_key=awsSecretAccessKeyOverride_)
            bucket = conn.get_bucket(bucketName)
            for kv in kvIter:
                idx, keyName = kv
                key = bucket.get_key(keyName)
                buf = key.get_contents_as_string()
                yield idx, buf

        self.lastNRecs = len(keyNameList)
        npartitions = min(npartitions, self.lastNRecs) if npartitions else self.lastNRecs
        return self.sc.parallelize(enumerate(keyNameList), npartitions).mapPartitions(readSplitFromS3)


class LocalFSFileReader(object):
    """File reader backed by python's native file() objects.
    """
    def __init__(self, **kwargs):
        # do nothing; allows AWS access keys to be passed in to a generic Reader instance w/o blowing up
        pass

    def __listRecursive(self, dataPath):
        if os.path.isdir(dataPath):
            dirname = dataPath
            matchpattern = None
        else:
            dirname, matchpattern = os.path.split(dataPath)

        filenames = set()
        for root, dirs, files in os.walk(dirname):
            if matchpattern:
                files = fnmatch.filter(files, matchpattern)
            for filename in files:
                filenames.add(os.path.join(root, filename))
        filenames = list(filenames)
        filenames.sort()
        return filenames

    def list(self, dataPath, filename=None, startIdx=None, stopIdx=None, recursive=False,
             includeDirectories=False):
        """List files specified by dataPath.

        Datapath may include a single wildcard ('*') in the filename specifier.

        Returns sorted list of absolute path strings.
        """
        absPath = LocalFSParallelReader.uriToPath(dataPath)

        if (not filename) and recursive:
            return self.__listRecursive(absPath)

        if filename:
            if os.path.isdir(absPath):
                absPath = os.path.join(absPath, filename)
            else:
                absPath = os.path.join(os.path.dirname(absPath), filename)
        else:
            if os.path.isdir(absPath) and not includeDirectories:
                absPath = os.path.join(absPath, "*")

        files = glob.glob(absPath)
        # filter out directories
        if not includeDirectories:
            files = [fpath for fpath in files if not os.path.isdir(fpath)]
        files.sort()
        files = selectByStartAndStopIndices(files, startIdx, stopIdx)
        return files

    def read(self, dataPath, filename=None, startOffset=None, size=-1):
        filenames = self.list(dataPath, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % dataPath)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % dataPath)

        return _localRead(filenames[0], startOffset=startOffset, size=size)

    def open(self, dataPath, filename=None):
        filenames = self.list(dataPath, filename=filename)

        if not filenames:
            raise FileNotFoundError("No file found matching: '%s'" % dataPath)
        if len(filenames) > 1:
            raise ValueError("Found multiple files matching: '%s'" % dataPath)

        return open(filenames[0], 'rb')


class BotoS3FileReader(_BotoS3Client):
    """File reader backed by the boto AWS client library.
    """
    def __getMatchingKeys(self, dataPath, filename=None):
        parse = _BotoS3Client.parseS3Query(dataPath)
        conn = boto.connect_s3(**AWSCredentials.getCredentialsAsDict(self.awsCredentialsOverride))
        bucketName = parse[0]
        keyName = parse[1]
        bucket = conn.get_bucket(bucketName)

        if filename:
            # check whether last section of dataPath refers to a directory
            if not keyName.endswith("/"):
                if self.checkPrefix(bucket, keyName + "/"):
                    # keyname is a directory, but we've omitted the trailing "/"
                    keyName += "/"
                else:
                    # assume keyname refers to an object other than a directory
                    # look for filename in same directory as keyname
                    slashIdx = keyName.rfind("/")
                    if slashIdx >= 0:
                        keyName = keyName[:(slashIdx+1)]
                    else:
                        # no directory separators, so our object is in the top level of the bucket
                        keyName = ""
            keyName += filename

        return _BotoS3Client.retrieveKeys(bucket, keyName, prefix=parse[2], postfix=parse[3])

    def list(self, dataPath, filename=None, startIdx=None, stopIdx=None, recursive=False, includeDirectories=True):
        """List s3 objects specified by dataPath.

        Returns sorted list of 's3n://' URIs.
        """
        # TODO: note that the default value for includeDirectories is here True, which reflects the current (and
        # longstanding) behavior of this class. This *differs* from the local FS list() method, which by default
        # filters out directories (which is definitely what we want if we're reading with recursive=True).
        # These two need to be made more consistent.
        if recursive:
            raise NotImplementedError("Recursive traversal of directories isn't yet implemented for S3 - sorry!")
        if not includeDirectories:
            raise NotImplementedError("Filtering out directories is not yet implemented for S3 - sorry!")

        keys = self.__getMatchingKeys(dataPath, filename=filename)
        keyNames = ["s3n:///" + key.bucket.name + "/" + key.name for key in keys]
        keyNames.sort()
        keyNames = selectByStartAndStopIndices(keyNames, startIdx, stopIdx)
        return keyNames

    def __getSingleMatchingKey(self, dataPath, filename=None):
        keys = self.__getMatchingKeys(dataPath, filename=filename)
        # keys is probably a lazy-loading ifilter iterable
        try:
            key = keys.next()
        except StopIteration:
            raise FileNotFoundError("Could not find S3 object for: '%s'" % dataPath)

        # we expect to only have a single key returned
        nextKey = None
        try:
            nextKey = keys.next()
        except StopIteration:
            pass
        if nextKey:
            raise ValueError("Found multiple S3 keys for: '%s'" % dataPath)
        return key

    def read(self, dataPath, filename=None, startOffset=None, size=-1):
        key = self.__getSingleMatchingKey(dataPath, filename=filename)

        if startOffset or (size > -1):
            # specify Range header in S3 request
            # see: http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.35
            # and: http://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectGET.html
            if not startOffset:
                startOffset = 0
            if size > -1:
                sizeStr = startOffset + size - 1  # range header is inclusive
            else:
                sizeStr = ""
            hdrs = {"Range": "bytes=%d-%s" % (startOffset, sizeStr)}
            return key.get_contents_as_string(headers=hdrs)
        else:
            return key.get_contents_as_string()

    def open(self, dataPath, filename=None):
        key = self.__getSingleMatchingKey(dataPath, filename=filename)
        return BotoS3ReadFileHandle(key)


class BotoS3ReadFileHandle(object):
    """Read-only file handle-like object exposing a subset of file methods.

    Returned by BotoS3FileReader's open() method.
    """
    def __init__(self, key):
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
        return "s3n:///" + self._key.bucket.name + "/" + self._key.name

    @property
    def mode(self):
        return "rb"


SCHEMAS_TO_PARALLELREADERS = {
    '': LocalFSParallelReader,
    'file': LocalFSParallelReader,
    's3': BotoS3ParallelReader,
    's3n': BotoS3ParallelReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}

SCHEMAS_TO_FILEREADERS = {
    '': LocalFSFileReader,
    'file': LocalFSFileReader,
    's3': BotoS3FileReader,
    's3n': BotoS3FileReader,
    'hdfs': None,
    'http': None,
    'https': None,
    'ftp': None
}


def getByScheme(dataPath, lookup, default):
    """Helper function used by get*ForPath().
    """
    parseresult = urlparse.urlparse(dataPath)
    clazz = lookup.get(parseresult.scheme, default)
    if clazz is None:
        raise NotImplementedError("No implementation for scheme " + parseresult.scheme)
    return clazz


def getParallelReaderForPath(dataPath):
    """Returns the class of a parallel reader suitable for the scheme used by `dataPath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSParallelReader if scheme is absent or not recognized.
    """
    return getByScheme(dataPath, SCHEMAS_TO_PARALLELREADERS, LocalFSParallelReader)


def getFileReaderForPath(dataPath):
    """Returns the class of a file reader suitable for the scheme used by `dataPath`.

    The resulting class object must still be instantiated in order to get a usable instance of the class.

    Throws NotImplementedError if the requested scheme is explicitly not supported (e.g. "ftp://").
    Returns LocalFSFileReader if scheme is absent or not recognized.
    """
    return getByScheme(dataPath, SCHEMAS_TO_FILEREADERS, LocalFSFileReader)
