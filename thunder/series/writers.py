from numpy import prod, unravel_index

def tobinary(series, path, prefix='series', overwrite=False, credentials=None):
    """
    Writes out data to binary format.

    Parameters
    ----------
    series : Series
        The data to write

    path : string path or URI to directory to be created
        Output files will be written underneath path.
        Directory will be created as a result of this call.

    prefix : str, optional, default = 'series'
        String prefix for files.

    overwrite : bool
        If true, path and all its contents will be deleted and
        recreated as partof this call.
    """
    from six import BytesIO
    from thunder.utils import check_path
    from thunder.writers import get_parallel_writer

    if not overwrite:
        check_path(path, credentials=credentials)
        overwrite = True

    def tobuffer(kv):
        firstkey = None
        buf = BytesIO()
        for k, v in kv:
            if firstkey is None:
                firstkey = k
            buf.write(v.tostring())
        val = buf.getvalue()
        buf.close()
        if firstkey is None:
            return iter([])
        else:
            label = prefix + '-' + getlabel(firstkey) + ".bin"
            return iter([(label, val)])

    writer = get_parallel_writer(path)(path, overwrite=overwrite, credentials=credentials)

    if series.mode == 'spark':
        binary = series.values.tordd().sortByKey().mapPartitions(tobuffer)
        binary.foreach(writer.write)

    else:
        basedims = [series.shape[d] for d in series.baseaxes]

        def split(k):
            ind = unravel_index(k, basedims)
            return ind, series.values[ind]

        buf = tobuffer([split(i) for i in range(prod(basedims))])
        [writer.write(b) for b in buf]

    shape = series.shape
    dtype = series.dtype

    write_config(path, shape=shape, dtype=dtype, overwrite=overwrite, credentials=credentials)

def write_config(path, shape=None, dtype=None, name="conf.json", overwrite=True, credentials=None):
    """
    Write a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.writers import get_file_writer

    writer = get_file_writer(path)
    conf = {'shape': shape, 'dtype': str(dtype)}

    confwriter = writer(path, name, overwrite=overwrite, credentials=credentials)
    confwriter.write(json.dumps(conf, indent=2))

    successwriter = writer(path, "SUCCESS", overwrite=overwrite, credentials=credentials)
    successwriter.write('')

def getlabel(key):
    """
    Get a file label from keys with reversed order
    """
    return '-'.join(["%05g" % k for k in key])
