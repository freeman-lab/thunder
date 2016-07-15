from numpy import array, arange, frombuffer, load, asarray, random, \
    fromstring, expand_dims, unravel_index, prod

try:
    buffer
except NameError:
    buffer = memoryview

from ..utils import check_spark, check_options
spark = check_spark()


def fromrdd(rdd, nrecords=None, shape=None, index=None, labels=None, dtype=None, ordered=False):
    """
    Load series data from a Spark RDD.

    Assumes keys are tuples with increasing and unique indices,
    and values are 1d ndarrays. Will try to infer properties
    that are not explicitly provided.

    Parameters
    ----------
    rdd : SparkRDD
        An RDD containing series data.

    shape : tuple or array, optional, default = None
        Total shape of data (if provided will avoid check).

    nrecords : int, optional, default = None
        Number of records (if provided will avoid check).

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    labels : array, optional, default = None
        Labels for records. If provided, should have shape of shape[:-1].

    dtype : string, default = None
        Data numerical type (if provided will avoid check)

    ordered : boolean, optional, default = False
        Whether or not the rdd is ordered by key
    """
    from .series import Series
    from bolt.spark.array import BoltArraySpark

    if index is None or dtype is None:
        item = rdd.values().first()

    if index is None:
        index = range(len(item))

    if dtype is None:
        dtype = item.dtype

    if nrecords is None and shape is not None:
        nrecords = prod(shape[:-1])

    if nrecords is None:
        nrecords = rdd.count()

    if shape is None:
        shape = (nrecords, asarray(index).shape[0])

    def process_keys(record):
        k, v = record
        if isinstance(k, int):
            k = (k,)
        return k, v

    values = BoltArraySpark(rdd.map(process_keys), shape=shape, dtype=dtype, split=len(shape)-1, ordered=ordered)
    return Series(values, index=index, labels=labels)

def fromarray(values, index=None, labels=None, npartitions=None, engine=None):
    """
    Load series data from an array.

    Assumes that all but final dimension index the records,
    and the size of the final dimension is the length of each record,
    e.g. a (2, 3, 4) array will be treated as 2 x 3 records of size (4,)

    Parameters
    ----------
    values : array-like
        An array containing the data. Can be a numpy array,
        a bolt array, or an array-like.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    labels : array, optional, default = None
        Labels for records. If provided, should have same shape as values.shape[:-1].

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    from .series import Series
    import bolt

    if isinstance(values, bolt.spark.array.BoltArraySpark):
        return Series(values)

    values = asarray(values)

    if values.ndim < 2:
        values = expand_dims(values, 0)

    if index is not None and not asarray(index).shape[0] == values.shape[-1]:
        raise ValueError('Index length %s not equal to record length %s'
                         % (asarray(index).shape[0], values.shape[-1]))
    if index is None:
        index = arange(values.shape[-1])

    if spark and isinstance(engine, spark):
        axis = tuple(range(values.ndim - 1))
        values = bolt.array(values, context=engine, npartitions=npartitions, axis=axis)
        values._ordered = True
        return Series(values, index=index)

    return Series(values, index=index, labels=labels)

def fromlist(items, accessor=None, index=None, labels=None, dtype=None, npartitions=None, engine=None):
    """
    Load series data from a list with an optional accessor function.

    Will call accessor function on each item from the list,
    providing a generic interface for data loading.

    Parameters
    ----------
    items : list
        A list of items to load.

    accessor : function, optional, default = None
        A function to apply to each item in the list during loading.

    index : array, optional, default = None
        Index for records, if not provided will use (0,1,...,N)
        where N is the length of each record.

    labels : array, optional, default = None
        Labels for records. If provided, should have same length as items.

    dtype : string, default = None
        Data numerical type (if provided will avoid check)

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    if spark and isinstance(engine, spark):
        if dtype is None:
            dtype = accessor(items[0]).dtype if accessor else items[0].dtype
        nrecords = len(items)
        keys = map(lambda k: (k, ), range(len(items)))
        if not npartitions:
            npartitions = engine.defaultParallelism
        items = zip(keys, items)
        rdd = engine.parallelize(items, npartitions)
        if accessor:
            rdd = rdd.mapValues(accessor)
        return fromrdd(rdd, nrecords=nrecords, index=index, labels=labels, dtype=dtype, ordered=True)

    else:
        if accessor:
            items = [accessor(i) for i in items]
        return fromarray(items, index=index, labels=labels)

def fromtext(path, ext='txt', dtype='float64', skip=0, shape=None, index=None, labels=None, npartitions=None, engine=None, credentials=None):
    """
    Loads series data from text files.

    Assumes data are formatted as rows, where each record is a row
    of numbers separated by spaces e.g. 'v v v v v'. You can
    optionally specify a fixed number of initial items per row to skip / discard.

    Parameters
    ----------
    path : string
        Directory to load from, can be a URI string with scheme
        (e.g. 'file://', 's3n://', or 'gs://'), or a single file,
        or a directory, or a directory with a single wildcard character.

    ext : str, optional, default = 'txt'
        File extension.

    dtype : dtype or dtype specifier, default 'float64'
        Numerical type to use for data after converting from text.

    skip : int, optional, default = 0
        Number of items in each record to skip.

    shape : tuple or list, optional, default = None
        Shape of data if known, will be inferred otherwise.

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    labels : array, optional, default = None
        Labels for records. If provided, should have length equal to number of rows.

    npartitions : int, default = None
        Number of partitions for parallelization (Spark only)

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)

    credentials : dict, default = None
        Credentials for remote storage (e.g. S3) in the form {access: ***, secret: ***}
    """
    from thunder.readers import normalize_scheme, get_parallel_reader
    path = normalize_scheme(path, ext)

    if spark and isinstance(engine, spark):

        def parse(line, skip):
            vec = [float(x) for x in line.split(' ')]
            return array(vec[skip:], dtype=dtype)

        lines = engine.textFile(path, npartitions)
        data = lines.map(lambda x: parse(x, skip))

        def switch(record):
            ary, idx = record
            return (idx,), ary

        rdd = data.zipWithIndex().map(switch)
        return fromrdd(rdd, dtype=str(dtype), shape=shape, index=index, ordered=True)

    else:
        reader = get_parallel_reader(path)(engine, credentials=credentials)
        data = reader.read(path, ext=ext)

        values = []
        for kv in data:
            for line in str(kv[1].decode('utf-8')).split('\n')[:-1]:
                values.append(fromstring(line, sep=' '))
        values = asarray(values)

        if skip > 0:
            values = values[:, skip:]

        if shape:
            values = values.reshape(shape)

        return fromarray(values, index=index, labels=labels)

def frombinary(path, ext='bin', conf='conf.json', dtype=None, shape=None, skip=0, index=None, labels=None, engine=None, credentials=None):
    """
    Load series data from flat binary files.

    Parameters
    ----------
    path : string URI or local filesystem path
        Directory to load from, can be a URI string with scheme
        (e.g. 'file://', 's3n://', or 'gs://'), or a single file,
        or a directory, or a directory with a single wildcard character.

    ext : str, optional, default = 'bin'
        Optional file extension specifier.

    conf : str, optional, default = 'conf.json'
        Name of conf file with type and size information.

    dtype : dtype or dtype specifier, default 'float64'
        Numerical type to use for data after converting from text.

    shape : tuple or list, optional, default = None
        Shape of data if known, will be inferred otherwise.

    skip : int, optional, default = 0
        Number of items in each record to skip.

    index : array, optional, default = None
        Index for records, if not provided will use (0, 1, ...)

    labels : array, optional, default = None
        Labels for records. If provided, should have shape of shape[:-1].

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)

    credentials : dict, default = None
        Credentials for remote storage (e.g. S3) in the form {access: ***, secret: ***}
    """
    shape, dtype = _binaryconfig(path, conf, dtype, shape, credentials)

    from thunder.readers import normalize_scheme, get_parallel_reader
    path = normalize_scheme(path, ext)

    from numpy import dtype as dtype_func
    nelements = shape[-1] + skip
    recordsize = dtype_func(dtype).itemsize * nelements

    if spark and isinstance(engine, spark):
        lines = engine.binaryRecords(path, recordsize)
        raw = lines.map(lambda x: frombuffer(buffer(x), offset=0, count=nelements, dtype=dtype)[skip:])

        def switch(record):
            ary, idx = record
            return (idx,), ary

        rdd = raw.zipWithIndex().map(switch)

        if shape and len(shape) > 2:
            expand = lambda k: unravel_index(k[0], shape[0:-1])
            rdd = rdd.map(lambda kv: (expand(kv[0]), kv[1]))

        if not index:
            index = arange(shape[-1])

        return fromrdd(rdd, dtype=dtype, shape=shape, index=index, ordered=True)

    else:
        reader = get_parallel_reader(path)(engine, credentials=credentials)
        data = reader.read(path, ext=ext)

        values = []
        for record in data:
            buf = record[1]
            offset = 0
            while offset < len(buf):
                v = frombuffer(buffer(buf), offset=offset, count=nelements, dtype=dtype)
                values.append(v[skip:])
                offset += recordsize

        if not len(values) == prod(shape[0:-1]):
            raise ValueError('Unexpected shape, got %g records but expected %g'
                             % (len(values), prod(shape[0:-1])))

        values = asarray(values, dtype=dtype)

        if shape:
            values = values.reshape(shape)

        return fromarray(values, index=index, labels=labels)

def _binaryconfig(path, conf, dtype=None, shape=None, credentials=None):
    """
    Collects parameters to use for binary series loading.
    """
    import json
    from thunder.readers import get_file_reader, FileNotFoundError

    reader = get_file_reader(path)(credentials=credentials)
    try:
        buf = reader.read(path, filename=conf)
        params = json.loads(str(buf.decode('utf-8')))
    except FileNotFoundError:
        params = {}

    if dtype:
        params['dtype'] = dtype

    if shape:
        params['shape'] = shape

    if 'dtype' not in params.keys():
        raise ValueError('dtype not specified either in conf.json or as argument')

    if 'shape' not in params.keys():
        raise ValueError('shape not specified either in conf.json or as argument')

    return params['shape'], params['dtype']

def fromrandom(shape=(100, 10), npartitions=1, seed=42, engine=None):
    """
    Generate random gaussian series data.

    Parameters
    ----------
    shape : tuple, optional, default = (100,10)
        Dimensions of data.

    npartitions : int, optional, default = 1
        Number of partitions with which to distribute data.

    seed : int, optional, default = 42
        Randomization seed.

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    seed = hash(seed)

    def generate(v):
        random.seed(seed + v)
        return random.randn(shape[1])

    return fromlist(range(shape[0]), accessor=generate, npartitions=npartitions, engine=engine)

def fromexample(name=None, engine=None):
    """
    Load example series data.

    Data are downloaded from S3, so this method requires an internet connection.

    Parameters
    ----------
    name : str
        Name of dataset, options include 'iris' | 'mouse' | 'fish'.
        If not specified will print options.

    engine : object, default = None
        Computational engine (e.g. a SparkContext for Spark)
    """
    import os
    import tempfile
    import shutil
    from boto.s3.connection import S3Connection

    datasets = ['iris', 'mouse', 'fish']

    if name is None:
        print('Availiable example series datasets')
        for d in datasets:
            print('- ' + d)
        return

    check_options(name, datasets)

    d = tempfile.mkdtemp()

    try:
        os.mkdir(os.path.join(d, 'series'))
        os.mkdir(os.path.join(d, 'series', name))
        conn = S3Connection(anon=True)
        bucket = conn.get_bucket('thunder-sample-data')
        for key in bucket.list(os.path.join('series', name) + '/'):
            if not key.name.endswith('/'):
                key.get_contents_to_filename(os.path.join(d, key.name))
        data = frombinary(os.path.join(d, 'series', name), engine=engine)

        if spark and isinstance(engine, spark):
            data.cache()
            data.compute()

    finally:
        shutil.rmtree(d)

    return data
