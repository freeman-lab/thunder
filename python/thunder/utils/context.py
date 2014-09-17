""" Simple wrapper for a Spark Context to provide loading functionality """

import glob
import os
from numpy import int16, float, dtype, frombuffer, zeros, fromfile, \
    asarray, mod, floor, ceil, shape, concatenate, prod, arange
from scipy.io import loadmat
from pyspark import SparkContext
import json
from thunder.utils.load import PreProcessor, Parser, indtosub
from thunder.utils import DataSets


class ThunderContext():
    """
    Wrapper for a Spark Context

    Attributes
    ----------
    `_sc` : SparkContext
        Spark context for Spark functionality
    """

    def __init__(self, sparkcontext):
        self._sc = sparkcontext

    @classmethod
    def start(cls, *args, **kwargs):
        """Starts a ThunderContext using the same arguments as SparkContext"""
        return ThunderContext(SparkContext(*args, **kwargs))

    def loadText(self, datafile, nkeys=3, filter=None, minPartitions=None):
        """
        Load data from text file (or a directory of files) with rows as
        <k1> <k2> ... <v1> <v2> ...
        where <k1> <k2> ... are keys and <v1> <v2> ... are the data values
        and rows are separated by line breaks
        Files can be local or stored on HDFS / S3
        If passed a directory, will automatically sort files by name

        Parameters
        ----------
        datafile : str
            Location of file(s)

        nkeys : int, optional, default = 3
            Number of keys per data point

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        npartitions : int, optional, default = None
            Number of partitions

        Returns
        -------
        data : RDD of (tuple, array) pairs
            The parsed and preprocessed data as an RDD
        """

        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.txt')))
            datafile = ''.join([files[x] + ',' for x in range(0, len(files))])[0:-1]

        lines = self._sc.textFile(datafile, minPartitions)
        parser = Parser(nkeys)
        data = lines.map(parser.get)

        return preprocess(data, method=filter)

    def loadBinary(self, datafile, nvalues=None, format='int16', nkeys=3, filter=None):
        """
        Load data from flat binary file (or a directory of files) with format
        <k1> <k2> ... <v1> <v2> ... <k1> <k2> ... <v1> <v2> ...
        where <k1> <k2> ... are keys and <v1> <v2> ... are data values
        Each record must contain the same total number of keys and values
        If nkeys is 0, a single index key will be added to each record
        Files can be local or stored on HDFS / S3
        Data parameters (number of values, number of keys, and format) can be
        given as keyword arguments, or provided in a JSON configuration file.
        Without a configuration file, must provide at least the number of values.

        Parameters
        ----------
        datafile : str
            Location of file(s)

        nvalues : int, optional, default = None
            Number of values per record

        nkeys : int, optional, default = 3
            Number of keys per record

        format : string, optional, default = 'int16'
            Format to use when parsing binary data, string specification
            of a numpy.dtype

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        Returns
        -------
        data : RDD of (tuple, array) pairs
            The parsed and preprocessed data as an RDD
        """

        if os.path.isdir(datafile):
            basepath = datafile
            datafile = os.path.join(basepath, '*.bin')
        else:
            basepath = os.path.dirname(datafile)

        try:
            f = open(os.path.join(basepath, 'conf.json'), 'r')
            params = json.load(f)
            nvalues = params['nvalues']
            nkeys = params['nkeys']
            format = params['format']
        except IOError:
            if nvalues is None:
                raise StandardError('must specify nvalues if there is no configuration file')

        recordsize = nvalues + nkeys
        recordsize *= dtype(FORMATS[format]).itemsize

        lines = self._sc.newAPIHadoopFile(datafile, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                          'org.apache.hadoop.io.LongWritable',
                                          'org.apache.hadoop.io.BytesWritable',
                                          conf={'recordLength': str(recordsize)})

        parsed = lines.map(lambda (k, v): (k, frombuffer(v, format)))
        data = parsed.map(lambda (k, v): (tuple(v[0:nkeys].astype(int)), v[nkeys:].astype(float)))

        return preprocess(data, method=filter)

    def makeExample(self, dataset, **opts):
        """
        Make an example data set for testing analyses
        see DataSets

        Parameters
        ----------
        dataset : str
            Which dataset to generate

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        return DataSets.make(self._sc, dataset, **opts)

    def loadExample(self, dataset):
        """
        Load a local example data set for testing analyses

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset
        """

        path = os.path.dirname(os.path.realpath(__file__))

        if dataset == "iris":
            return self.loadText(os.path.join(path, 'data/iris.txt'), minPartitions=1)
        elif dataset == "fish":
            return self.loadText(os.path.join(path, 'data/fish.txt'), minPartitions=1)
        else:
            raise NotImplementedError("dataset '%s' not found" % dataset)

    def loadExampleEC2(self, dataset):
        """
        Load an example data set from EC2

        Parameters
        ----------
        dataset : str
            Which dataset to load

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Generated dataset

        params : Tuple or numpy array
            Parameters or metadata for dataset
        """

        if 'ec' not in self._sc.master:
            raise Exception("must be running on EC2 to load this example data sets")
        elif dataset == "zebrafish-optomotor-response":
            path = 'zebrafish.datasets/optomotor-response/1/'
            data = self.loadText("s3n://" + path + 'data/dat_plane*.txt', filter='dff', minPartitions=1000)
            paramfile = self._sc.textFile("s3n://" + path + "params.json")
            params = json.loads(paramfile.first())
            modelfile = asarray(params['trials'])
            return data, modelfile
        else:
            raise NotImplementedError("dataset '%s' not availiable" % dataset)

    def convertStacks(self, datafile, dims, savefile, nblocks=None, filerange=None):
        """
        Convert data from binary stack files to reformatted flat binary files,
        see also convertStack

        Currently only supported on a local or networked file system

        Parameters
        ----------
        datafile : str
            File(s) or directory to convert

        dims : list
            Stack dimensions

        savefile : str
            Location to save the converted data

        nblocks : int, optional, default = None (automatically set)
            Number of blocks to split data into

        filerange : list, optional, default = None (all files)
            Indices of first and last file to include

        """
        rdd = self.importStacksAsBlocks(datafile, dims, nblocks=nblocks, filerange=filerange)

        # save blocks of data to flat binary files
        def writeblock(part, mat, path):
            filename = os.path.join(path, "part-" + str(part) + ".bin")
            mat.tofile(filename)

        if os.path.isdir(savefile):
            raise IOError('path %s already exists' % savefile)
        else:
            os.mkdir(savefile)

        rdd.foreach(lambda (ip, mat): writeblock(ip, mat, savefile))

        # write configuration file
        if not filerange:
            if os.path.isdir(datafile):
                files = glob.glob(os.path.join(datafile, '*.stack'))
            else:
                files = glob.glob(datafile)
            filerange = [0, len(files)-1]
        logout = {'input': datafile, 'filerange': filerange, 'dims': dims,
                  'nkeys': len(dims), 'nvalues': filerange[1]-filerange[0]+1, 'format': 'int16'}
        f = open(os.path.join(savefile, 'conf.json'), 'w')
        json.dump(logout, f, indent=2)
        f.close()

        # write SUCCESS file
        f = open(os.path.join(savefile, 'SUCCESS'), 'w')
        f.write(' ')
        f.close()

    def importStacks(self, datafile, dims, nblocks=None, filerange=None, filter=None):
        """
        Import data from binary stack files as an RDD,
        see also convertStack

        Currently only supported on a local or networked file system

        Parameters
        ----------
        datafile : str
            File(s) or directory to import

        dims : list
            Stack dimensions

        nblocks : int, optional, default = None (automatically set)
            Number of blocks to split data into

        filerange : list, optional, default = None (all files)
            Indices of first and last file to include

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        Returns
        -------
        data : RDD of (tuple, array) pairs
            Parsed and preprocessed data
        """
        rdd = self.importStacksAsBlocks(datafile, dims, nblocks=nblocks, filerange=filerange)
        nkeys = len(dims)
        data = rdd.values().flatMap(lambda x: list(x)).map(lambda x: (tuple(x[0:nkeys].astype(int)), x[nkeys:]))
        return preprocess(data, method=filter)

    def importStacksAsBlocks(self, datafile, dims, nblocks=None, filerange=None):
        """
        Convert data from binary stack files to blocks of an RDD,
        which can either be saved to flat binary files,
        or returned as an flattened RDD (see convertStack and importStack)

        Stacks are typically flat binary files containing
        2-dimensional or 3-dimensional image data

        We assume there are multiple files:

        file0.stack, file1.stack, file2.stack, ...

        This function loads the same contiguous block from all files,
        and rewrites the result to flat binary files of the form:

        block0.bin, block1.bin, block2.bin, ...

        Currently only supported on a local or networked file system

        TODO: Add support for EC2 loading
        TODO: assumes int16, add support for other formats

        """

        # get the paths to the data
        if os.path.isdir(datafile):
            files = sorted(glob.glob(os.path.join(datafile, '*.stack')))
        else:
            files = sorted(glob.glob(datafile))
        if len(files) < 1:
            raise IOError('cannot find any stack files in %s' % datafile)
        if filerange:
            files = files[filerange[0]:filerange[1]+1]

        # get the total stack dimensions
        totaldim = float(prod(dims))

        # if number of blocks not provided, start by setting it
        # so that each block is approximately 200 MB
        if not nblocks:
            nblocks = int(max(floor((totaldim * len(files) * 2) / (200 * 10**6)), 1))

        if len(dims) == 3:
            # for 3D stacks, do calculations to ensure that
            # different planes appear in distinct files
            k = max(int(floor(float(nblocks) / dims[2])), 1)
            n = dims[0] * dims[1]
            kupdated = [x for x in range(1, k+1) if mod(n, x) == 0][-1]
            nblocks = kupdated * dims[2]
            blocksize = int(totaldim / nblocks)
        else:
            # otherwise just round to make contents divide into nearly even blocks
            blocksize = int(ceil(totaldim / float(nblocks)))
            nblocks = int(ceil(totaldim / float(blocksize)))

        def readblock(block, files, blocksize):
            # get start position for this block
            position = block * blocksize

            # adjust if at end of file
            if (position + blocksize) > totaldim:
                blocksize = int(totaldim - position)

            # loop over files, loading one block from each
            mat = zeros((blocksize, len(files)))

            for i, f in enumerate(files):
                fid = open(f, "rb")
                fid.seek(position * dtype(int16).itemsize)
                mat[:, i] = fromfile(fid, dtype=int16, count=blocksize)

            # append subscript keys based on dimensions
            lininds = range(position + 1, position + shape(mat)[0] + 1)
            keys = asarray(map(lambda (k, v): k, indtosub(zip(lininds, zeros(blocksize)), dims)))
            partlabel = "%05g-" % block + (('%05g-' * len(dims)) % tuple(keys[0]))[:-1]
            return partlabel, concatenate((keys, mat), axis=1).astype(int16)

        # map over blocks
        blocks = range(0, nblocks)
        return self._sc.parallelize(blocks, len(blocks)).map(lambda b: readblock(b, files, blocksize))

    def loadBinaryLocal(self, datafile, nvalues, nkeys, format, keyfile=None, method=None):
        """
        Load data from a local binary file
        """

        raise NotImplementedError

    def loadArrayLocal(self, values, keys=None, method=None):
        """
        Load data from local arrays
        """

        raise NotImplementedError

    def loadMatLocal(self, datafile, varname, keyfile=None, filter=None, minPartitions=1):
        """
        Load data from a local MAT file, from a variable containing
        either a 1d or 2d matrix, into an RDD of (key,value) pairs.
        Each row of the input matrix will become the value of each record.

        Keys can be provided in an extra MAT file containing a variable 'keys'.
        If not provided, linear indices will be used as keys.

        Parameters
        ----------
        datafile : str
            MAT file to import

        varname : str
            Variable name to load from MAT file

        keyfile : str
            MAT file to import with keys (must contain a variable 'keys')

        filter : str, optional, default = None (no preprocessing)
            Which preprocessing to perform

        minPartitions : Int, optional, default = 1
            Number of partitions for data

        """

        data = loadmat(datafile)[varname]
        if data.ndim > 2:
            raise IOError('input data must be one or two dimensional')
        if keyfile:
            keys = map(lambda x: tuple(x), loadmat(keyfile)['keys'])
        else:
            keys = arange(1, shape(data)[0]+1)

        rdd = self._sc.parallelize(zip(keys, data), minPartitions)

        return preprocess(rdd, method=filter)


def preprocess(data, method=None):

    if method:
        preprocessor = PreProcessor(method)
        return data.mapValues(preprocessor.get)
    else:
        return data

FORMATS = {
    'int16': int16,
    'float': float
}