import itertools
import logging
from io import BytesIO
from numpy import frombuffer, prod, random, asarray, expand_dims

from ..utils import check_spark, check_options
spark = check_spark()


def fromrdd(rdd, dims=None, nrecords=None, dtype=None, labels=None, ordered=False):
    """
    Load images from a Spark RDD.

    Input RDD must be a collection of key-value pairs
    where keys are singleton tuples indexing images,
    and values are 2d or 3d ndarrays.

    Parameters
    ----------
    rdd : SparkRDD
        An RDD containing the images.

    dims : tuple or array, optional, default = None
        Image dimensions (if provided will avoid check).

    nrecords : int, optional, default = None
        Number of images (if provided will avoid check).

    dtype : string, default = None
        Data numerical type (if provided will avoid check)

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.

    ordered : boolean, optional, default = False
        Whether or not the rdd is ordered by key
    """
    from .images import Images
    from bolt.spark.array import BoltArraySpark

    if dims is None or dtype is None:
        item = rdd.values().first()
        dtype = item.dtype
        dims = item.shape

    if nrecords is None:
        nrecords = rdd.count()

    def process_keys(record):
        k, v = record
        if isinstance(k, int):
            k = (k,)
        return k, v

    values = BoltArraySpark(rdd.map(process_keys), shape=(nrecords,) + tuple(dims), dtype=dtype, split=1, ordered=ordered)
    return Images(values, labels=labels)

def fromarray(values, labels=None, npartitions=None, engine=None):
    """
    Load images from an array.

    First dimension will be used to index images,
    so remaining dimensions after the first should
    be the dimensions of the images,
    e.g. (3, 100, 200) for 3 x (100, 200) images

    Parameters
    ----------
    values : array-like
        The array of images. Can be a numpy array,
        a bolt array, or an array-like.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.

    npartitions : int, default = None
        Number of partitions for parallelization (spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for spark)
    """
    from .images import Images
    import bolt

    if isinstance(values, bolt.spark.array.BoltArraySpark):
        return Images(values)

    values = asarray(values)

    if values.ndim < 2:
        raise ValueError('Array for images must have at least 2 dimensions, got %g' % values.ndim)

    if values.ndim == 2:
        values = expand_dims(values, 0)

    shape = None
    dtype = None
    for im in values:
        if shape is None:
            shape = im.shape
            dtype = im.dtype
        if not im.shape == shape:
            raise ValueError('Arrays must all be of same shape; got both %s and %s' %
                             (str(shape), str(im.shape)))
        if not im.dtype == dtype:
            raise ValueError('Arrays must all be of same data type; got both %s and %s' %
                             (str(dtype), str(im.dtype)))

    if spark and isinstance(engine, spark):
        if not npartitions:
            npartitions = engine.defaultParallelism
        values = bolt.array(values, context=engine, npartitions=npartitions, axis=(0,))
        values._ordered = True
        return Images(values)

    return Images(values, labels=labels)


def fromlist(items, accessor=None, keys=None, dims=None, dtype=None, labels=None, npartitions=None, engine=None):
    """
    Load images from a list of items using the given accessor.

    Parameters
    ----------
    accessor : function
        Apply to each item from the list to yield an image.

    keys : list, optional, default=None
        An optional list of keys.

    dims : tuple, optional, default=None
        Specify a known image dimension to avoid computation.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.

    npartitions : int
        Number of partitions for computational engine.
    """
    if spark and isinstance(engine, spark):
        nrecords = len(items)
        if keys:
            items = zip(keys, items)
        else:
            keys = [(i,) for i in range(nrecords)]
            items = zip(keys, items)
        if not npartitions:
            npartitions = engine.defaultParallelism
        rdd = engine.parallelize(items, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, nrecords=nrecords, dims=dims, dtype=dtype, labels=labels, ordered=True)

    else:
        if accessor:
            items = asarray([accessor(i) for i in items])
        return fromarray(items, labels=labels)

def frompath(path, accessor=None, ext=None, start=None, stop=None, recursive=False, npartitions=None, dims=None, dtype=None, labels=None, recount=False, engine=None, credentials=None):
    """
    Load images from a path using the given accessor.

    Supports both local and remote filesystems.

    Parameters
    ----------
    accessor : function
        Apply to each item after loading to yield an image.

    ext : str, optional, default=None
        File extension.

    npartitions : int, optional, default=None
        Number of partitions for computational engine,
        if None will use default for engine.

    dims : tuple, optional, default=None
        Dimensions of images.

    dtype : str, optional, default=None
        Numerical type of images.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.

    start, stop : nonnegative int, optional, default=None
        Indices of files to load, interpreted using Python slicing conventions.

    recursive : boolean, optional, default=False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    recount : boolean, optional, default=False
        Force subsequent record counting.
    """
    from thunder.readers import get_parallel_reader
    reader = get_parallel_reader(path)(engine, credentials=credentials)
    data = reader.read(path, ext=ext, start=start, stop=stop,
                       recursive=recursive, npartitions=npartitions)

    if spark and isinstance(engine, spark):
        if accessor:
            data = data.flatMap(accessor)
        if recount:
            nrecords = None

            def switch(record):
                ary, idx = record
                return (idx,), ary

            data = data.values().zipWithIndex().map(switch)
        else:
            nrecords = reader.nfiles
        return fromrdd(data, nrecords=nrecords, dims=dims, dtype=dtype, labels=labels, ordered=True)

    else:
        if accessor:
            data = [accessor(d) for d in data]
        flattened = list(itertools.chain(*data))
        values = [kv[1] for kv in flattened]
        return fromarray(values, labels=labels)


def frombinary(path, shape=None, dtype=None, ext='bin', start=None, stop=None, recursive=False, nplanes=None, npartitions=None, labels=None, conf='conf.json', order='C', engine=None, credentials=None):
    """
    Load images from flat binary files.

    Assumes one image per file, each with the shape and ordering as given
    by the input arguments.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    shape : tuple of positive int
        Dimensions of input image data.

    ext : string, optional, default = 'bin'
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default = None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching `path` and `ext`. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default = False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    nplanes : positive integer, optional, default = None
        If passed, will cause single files to be subdivided into nplanes separate images.
        Otherwise, each file is taken to represent one image.

    npartitions : int, optional, default = None
        Number of partitions for computational engine,
        if None will use default for engine.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.
    """
    import json
    from thunder.readers import get_file_reader, FileNotFoundError
    try:
        reader = get_file_reader(path)(credentials=credentials)
        buf = reader.read(path, filename=conf).decode('utf-8')
        params = json.loads(buf)
    except FileNotFoundError:
        params = {}

    if 'dtype' in params.keys():
        dtype = params['dtype']
    if 'dims' in params.keys():
        shape = params['dims']
    if 'shape' in params.keys():
        shape = params['shape']

    if not shape:
        raise ValueError('Image shape must be specified as argument or in a conf.json file')

    if not dtype:
        dtype = 'int16'

    if nplanes is not None:
        if nplanes <= 0:
            raise ValueError('nplanes must be positive if passed, got %d' % nplanes)
        if shape[-1] % nplanes:
            raise ValueError("Last dimension '%d' must be divisible by nplanes '%d'" %
                             (shape[-1], nplanes))

    def getarray(idx_buffer_filename):
        idx, buf, _ = idx_buffer_filename
        ary = frombuffer(buf, dtype=dtype, count=int(prod(shape))).reshape(shape, order=order)
        if nplanes is None:
            yield (idx,), ary
        else:
            # divide array into chunks of nplanes
            npoints = shape[-1] / nplanes  # integer division
            if shape[-1] % nplanes:
                npoints += 1
            timepoint = 0
            last_plane = 0
            current_plane = 1
            while current_plane < ary.shape[-1]:
                if current_plane % nplanes == 0:
                    slices = [slice(None)] * (ary.ndim - 1) + [slice(last_plane, current_plane)]
                    yield idx*npoints + timepoint, ary[slices].squeeze()
                    timepoint += 1
                    last_plane = current_plane
                current_plane += 1
            # yield remaining planes
            slices = [slice(None)] * (ary.ndim - 1) + [slice(last_plane, ary.shape[-1])]
            yield (idx*npoints + timepoint,), ary[slices].squeeze()

    recount = False if nplanes is None else True
    append = [nplanes] if (nplanes is not None and nplanes > 1) else []
    newdims = tuple(list(shape[:-1]) + append) if nplanes else shape
    return frompath(path, accessor=getarray, ext=ext, start=start,
                    stop=stop, recursive=recursive, npartitions=npartitions,
                    dims=newdims, dtype=dtype, labels=labels, recount=recount,
                    engine=engine, credentials=credentials)

def fromtif(path, ext='tif', start=None, stop=None, recursive=False, nplanes=None, npartitions=None, labels=None, engine=None, credentials=None, discard_extra=False):
    """
    Loads images from single or multi-page TIF files.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    ext : string, optional, default = 'tif'
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default = None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching 'path' and 'ext'. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default = False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    nplanes : positive integer, optional, default = None
        If passed, will cause single files to be subdivided into nplanes separate images.
        Otherwise, each file is taken to represent one image.

    npartitions : int, optional, default = None
        Number of partitions for computational engine,
        if None will use default for engine.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.

    discard_extra : boolean, optional, default = False
        If True and nplanes doesn't divide by the number of pages in a multi-page tiff, the reminder will
        be discarded and a warning will be shown. If False, it will raise an error
    """
    from tifffile import TiffFile

    if nplanes is not None and nplanes <= 0:
        raise ValueError('nplanes must be positive if passed, got %d' % nplanes)

    def getarray(idx_buffer_filename):
        idx, buf, fname = idx_buffer_filename
        fbuf = BytesIO(buf)
        tfh = TiffFile(fbuf)
        ary = tfh.asarray()
        pageCount = ary.shape[0]
        if nplanes is not None:
            extra = pageCount % nplanes
            if extra:
                if discard_extra:
                    pageCount = pageCount - extra
                    logging.getLogger('thunder').warn('Ignored %d pages in file %s' % (extra, fname))
                else:
                    raise ValueError("nplanes '%d' does not evenly divide '%d in file %s'" % (nplanes, pageCount,
                                                                                              fname))
            values = [ary[i:(i+nplanes)] for i in range(0, pageCount, nplanes)]
        else:
            values = [ary]
        tfh.close()

        if ary.ndim == 3:
            values = [val.squeeze() for val in values]

        nvals = len(values)
        keys = [(idx*nvals + timepoint,) for timepoint in range(nvals)]
        return zip(keys, values)

    recount = False if nplanes is None else True
    data = frompath(path, accessor=getarray, ext=ext, start=start, stop=stop,
                    recursive=recursive, npartitions=npartitions, recount=recount,
                    labels=labels, engine=engine, credentials=credentials)
    if engine is not None and npartitions is not None and data.npartitions() < npartitions:
        data = data.repartition(npartitions)
    return data

def frompng(path, ext='png', start=None, stop=None, recursive=False, npartitions=None, labels=None, engine=None, credentials=None):
    """
    Load images from PNG files.

    Parameters
    ----------
    path : str
        Path to data files or directory, specified as either a local filesystem path
        or in a URI-like format, including scheme. May include a single '*' wildcard character.

    ext : string, optional, default = 'tif'
        Extension required on data files to be loaded.

    start, stop : nonnegative int, optional, default = None
        Indices of the first and last-plus-one file to load, relative to the sorted
        filenames matching `path` and `ext`. Interpreted using python slice indexing conventions.

    recursive : boolean, optional, default = False
        If true, will recursively descend directories from path, loading all files
        with an extension matching 'ext'.

    npartitions : int, optional, default = None
        Number of partitions for computational engine,
        if None will use default for engine.

    labels : array, optional, default = None
        Labels for records. If provided, should be one-dimensional.
    """
    from scipy.misc import imread

    def getarray(idx_buffer_filename):
        idx, buf, _ = idx_buffer_filename
        fbuf = BytesIO(buf)
        yield (idx,), imread(fbuf)

    return frompath(path, accessor=getarray, ext=ext, start=start,
                    stop=stop, recursive=recursive, npartitions=npartitions,
                    labels=labels, engine=engine, credentials=credentials)

def fromrandom(shape=(10, 50, 50), npartitions=1, seed=42, engine=None):
    """
    Generate random image data.

    Parameters
    ----------
    shape : tuple, optional, default=(10, 50, 50)
        Dimensions of images.

    npartitions : int, optional, default=1
        Number of partitions.

    seed : int, optional, default=42
        Random seed.
    """
    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(*shape[1:])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions, engine=engine)

def fromexample(name=None, engine=None):
    """
    Load example image data.

    Data are downloaded from S3, so this method requires an internet connection.

    Parameters
    ----------
    name : str
        Name of dataset, if not specified will print options.

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    datasets = ['mouse', 'fish']

    if name is None:
        print('Availiable example image datasets')
        for d in datasets:
            print('- ' + d)
        return

    check_options(name, datasets)

    path = 's3n://thunder-sample-data/images/' + name

    if name == 'mouse':
        data = frombinary(path=path, npartitions=1, order='F', engine=engine)

    if name == 'fish':
        data = fromtif(path=path, npartitions=1, engine=engine)

    if spark and isinstance(engine, spark):
        data.cache()
        data.compute()

    return data
