"""Provides SeriesLoader object and helpers, used to read Series data from disk or other filesystems.
"""
from collections import namedtuple
import json
from numpy import array, arange, frombuffer, load, ndarray, unravel_index, vstack
from numpy import dtype as dtypeFunc
from scipy.io import loadmat
from cStringIO import StringIO
import itertools
import struct
import urlparse
import math

from thunder.rdds.fileio.writers import getParallelWriterForPath
from thunder.rdds.keys import Dimensions
from thunder.rdds.fileio.readers import getFileReaderForPath, FileNotFoundError, appendExtensionToPathSpec
from thunder.rdds.imgblocks.blocks import SimpleBlocks
from thunder.rdds.series import Series
from thunder.utils.common import parseMemoryString, smallestFloatType


class SeriesLoader(object):
    """Loader object used to instantiate Series data stored in a variety of formats.
    """
    def __init__(self, sparkContext, minPartitions=None,
                 awsCredentialsOverride=None):
        """Initialize a new SeriesLoader object.

        Parameters
        ----------
        sparkcontext: SparkContext
            The pyspark SparkContext object used by the current Thunder environment.

        minPartitions: int
            minimum number of partitions to use when loading data. (Used by fromText, fromMatLocal, and fromNpyLocal)
        """
        self.sc = sparkContext
        self.minPartitions = minPartitions
        self.awsCredentialsOverride = awsCredentialsOverride

    def fromArrays(self, arrays):
        """Create a Series object from a sequence of numpy ndarrays resident in memory on the driver.

        The arrays will be interpreted as though each represents a single time point - effectively the same
        as if converting Images to a Series, with each array representing a volume image at a particular
        point in time. Thus in the resulting Series, the value of the record with key (0,0,0) will be
        array([arrays[0][0,0,0], arrays[1][0,0,0],... arrays[n][0,0,0]).

        The dimensions of the resulting Series will be *opposite* that of the passed numpy array. Their dtype will not
        be changed.
        """
        # if passed a single array, cast it to a sequence of length 1
        if isinstance(arrays, ndarray):
            arrays = [arrays]

        # check that shapes of passed arrays are consistent
        shape = arrays[0].shape
        dtype = arrays[0].dtype
        for ary in arrays:
            if not ary.shape == shape:
                raise ValueError("Inconsistent array shapes: first array had shape %s, but other array has shape %s" %
                                 (str(shape), str(ary.shape)))
            if not ary.dtype == dtype:
                raise ValueError("Inconsistent array dtypes: first array had dtype %s, but other array has dtype %s" %
                                 (str(dtype), str(ary.dtype)))

        # get indices so that fastest index changes first
        shapeiters = (xrange(n) for n in shape)
        keys = [idx[::-1] for idx in itertools.product(*shapeiters)]

        values = vstack([ary.ravel() for ary in arrays]).T

        dims = Dimensions.fromTuple(shape[::-1])

        return Series(self.sc.parallelize(zip(keys, values), self.minPartitions), dims=dims, dtype=str(dtype),
                      awsCredentials=self.awsCredentialsOverride)

    @staticmethod
    def __normalizeDatafilePattern(dataPath, ext):
        dataPath = appendExtensionToPathSpec(dataPath, ext)
        # we do need to prepend a scheme here, b/c otherwise the Hadoop based readers
        # will adopt their default behavior and start looking on hdfs://.

        parseResult = urlparse.urlparse(dataPath)
        if parseResult.scheme:
            # this appears to already be a fully-qualified URI
            return dataPath
        else:
            # this looks like a local path spec
            # check whether we look like an absolute or a relative path
            import os
            dirComponent, fileComponent = os.path.split(dataPath)
            if not os.path.isabs(dirComponent):
                # need to make relative local paths absolute; our file scheme parsing isn't all that it could be.
                dirComponent = os.path.abspath(dirComponent)
                dataPath = os.path.join(dirComponent, fileComponent)
            return "file://" + dataPath

    def fromText(self, dataPath, nkeys=None, ext="txt", dtype='float64'):
        """
        Loads Series data from text files.

        Parameters
        ----------
        dataPath : string
            Specifies the file or files to be loaded. dataPath may be either a URI (with scheme specified) or a path
            on the local filesystem.
            If a path is passed (determined by the absence of a scheme component when attempting to parse as a URI),
            and it is not already a wildcard expression and does not end in <ext>, then it will be converted into a
            wildcard pattern by appending '/*.ext'. This conversion can be avoided by passing a "file://" URI.

        dtype: dtype or dtype specifier, default 'float64'

        """
        dataPath = self.__normalizeDatafilePattern(dataPath, ext)

        def parse(line, nkeys_):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys_:], dtype=dtype)
            keys = tuple(int(x) for x in vec[:nkeys_])
            return keys, ts

        lines = self.sc.textFile(dataPath, self.minPartitions)
        data = lines.map(lambda x: parse(x, nkeys))
        return Series(data, dtype=str(dtype), awsCredentials=self.awsCredentialsOverride)

    # keytype, valuetype here violate camelCasing convention for consistence with JSON conf file format
    BinaryLoadParameters = namedtuple('BinaryLoadParameters', 'nkeys nvalues keytype valuetype')
    BinaryLoadParameters.__new__.__defaults__ = (None, None, 'int16', 'int16')

    def __loadParametersAndDefaults(self, dataPath, confFilename, nkeys, nvalues, keyType, valueType):
        """Collects parameters to use for binary series loading.

        Priority order is as follows:
        1. parameters specified as keyword arguments;
        2. parameters specified in a conf.json file on the local filesystem;
        3. default parameters

        Returns
        -------
        BinaryLoadParameters instance
        """
        params = self.loadConf(dataPath, confFilename=confFilename)

        # filter dict to include only recognized field names:
        for k in params.keys():
            if k not in SeriesLoader.BinaryLoadParameters._fields:
                del params[k]
        keywordParams = {'nkeys': nkeys, 'nvalues': nvalues, 'keytype': keyType, 'valuetype': valueType}
        for k, v in keywordParams.items():
            if not v:
                del keywordParams[k]
        params.update(keywordParams)
        return SeriesLoader.BinaryLoadParameters(**params)

    @staticmethod
    def __checkBinaryParametersAreSpecified(paramsObj):
        """Throws ValueError if any of the field values in the passed namedtuple instance evaluate to False.

        Note this is okay only so long as zero is not a valid parameter value. Hmm.
        """
        missing = []
        for paramName, paramVal in paramsObj._asdict().iteritems():
            if not paramVal:
                missing.append(paramName)
        if missing:
            raise ValueError("Missing parameters to load binary series files - " +
                             "these must be given either as arguments or in a configuration file: " +
                             str(tuple(missing)))

    def fromBinary(self, dataPath, ext='bin', confFilename='conf.json',
                   nkeys=None, nvalues=None, keyType=None, valueType=None,
                   newDtype='smallfloat', casting='safe'):
        """
        Load a Series object from a directory of binary files.

        Parameters
        ----------

        dataPath: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

        newDtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        """

        paramsObj = self.__loadParametersAndDefaults(dataPath, confFilename, nkeys, nvalues, keyType, valueType)
        self.__checkBinaryParametersAreSpecified(paramsObj)

        dataPath = self.__normalizeDatafilePattern(dataPath, ext)

        keyDtype = dtypeFunc(paramsObj.keytype)
        valDtype = dtypeFunc(paramsObj.valuetype)

        keySize = paramsObj.nkeys * keyDtype.itemsize
        recordSize = keySize + paramsObj.nvalues * valDtype.itemsize

        lines = self.sc.newAPIHadoopFile(dataPath, 'thunder.util.io.hadoop.FixedLengthBinaryInputFormat',
                                         'org.apache.hadoop.io.LongWritable',
                                         'org.apache.hadoop.io.BytesWritable',
                                         conf={'recordLength': str(recordSize)})

        data = lines.map(lambda (_, v):
                         (tuple(int(x) for x in frombuffer(buffer(v, 0, keySize), dtype=keyDtype)),
                          frombuffer(buffer(v, keySize), dtype=valDtype)))

        return Series(data, dtype=str(valDtype), index=arange(paramsObj.nvalues),
                      awsCredentials=self.awsCredentialsOverride).astype(newDtype, casting)

    def _getSeriesBlocksFromStack(self, dataPath, dims, ext="stack", blockSize="150M", dtype='int16',
                                  newDtype='smallfloat', casting='safe', startIdx=None, stopIdx=None, recursive=False):
        """Create an RDD of <string blocklabel, (int k-tuple indices, array of datatype values)>

        Parameters
        ----------

        dataPath: string URI or local filesystem path
            Specifies the directory or files to be loaded. May be formatted as a URI string with scheme (e.g. "file://",
            "s3n://". If no scheme is present, will be interpreted as a path on the local filesystem. This path
            must be valid on all workers. Datafile may also refer to a single file, or to a range of files specified
            by a glob-style expression using a single wildcard character '*'.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        dtype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newDtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Series data must be floating-point. Input data will be cast to the
            requested `newdtype` - see numpy `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).

        Returns
        ---------
        pair of (RDD, ntimepoints)

        RDD: sequence of keys, values pairs
            (call using flatMap)

        RDD Key: tuple of int
            zero-based indicies of position within original image volume

        RDD Value: numpy array of datatype
            series of values at position across loaded image volumes

        ntimepoints: int
            number of time points in returned series, determined from number of stack files found at dataPath

        newDtype: string
            string representation of numpy data type of returned blocks

        """
        dataPath = self.__normalizeDatafilePattern(dataPath, ext)
        blockSize = parseMemoryString(blockSize)
        totalDim = reduce(lambda x_, y_: x_*y_, dims)
        dtype = dtypeFunc(dtype)
        if newDtype is None or newDtype == '':
            newDtype = str(dtype)
        elif newDtype == 'smallfloat':
            newDtype = str(smallestFloatType(dtype))
        else:
            newDtype = str(newDtype)

        reader = getFileReaderForPath(dataPath)(awsCredentialsOverride=self.awsCredentialsOverride)
        filenames = reader.list(dataPath, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)
        if not filenames:
            raise IOError("No files found for path '%s'" % dataPath)

        dataSize = totalDim * len(filenames) * dtype.itemsize
        nblocks = max(dataSize / blockSize, 1)  # integer division

        if len(dims) >= 3:
            # for 3D stacks, do calculations to ensure that
            # different planes appear in distinct files
            blocksPerPlane = max(nblocks / dims[-1], 1)

            pixPerPlane = reduce(lambda x_, y_: x_*y_, dims[:-1])  # all but last dimension

            # get the greatest number of blocks in a plane (up to as many as requested) that still divide the plane
            # evenly. This will always be at least one.
            kUpdated = [x for x in range(1, blocksPerPlane+1) if not pixPerPlane % x][-1]
            nblocks = kUpdated * dims[-1]
            blockSizePerStack = (totalDim / nblocks) * dtype.itemsize
        else:
            # otherwise just round to make contents divide into nearly even blocks
            blockSizePerStack = int(math.ceil(totalDim / float(nblocks)))
            nblocks = int(math.ceil(totalDim / float(blockSizePerStack)))
            blockSizePerStack *= dtype.itemsize

        fileSize = totalDim * dtype.itemsize

        def readBlock(blockNum):
            # copy size out from closure; will modify later:
            blockSizePerStack_ = blockSizePerStack
            # get start position for this block
            position = blockNum * blockSizePerStack_

            # adjust if at end of file
            if (position + blockSizePerStack_) > fileSize:
                blockSizePerStack_ = int(fileSize - position)
            # loop over files, loading one block from each
            bufs = []

            for fname in filenames:
                buf = reader.read(fname, startOffset=position, size=blockSizePerStack_)
                bufs.append(frombuffer(buf, dtype=dtype))

            buf = vstack(bufs).T  # dimensions are now linindex x time (images)
            del bufs
            buf = buf.astype(newDtype, casting=casting, copy=False)

            # append subscript keys based on dimensions
            itemPosition = position / dtype.itemsize
            itemBlocksize = blockSizePerStack_ / dtype.itemsize
            linearIdx = arange(itemPosition, itemPosition + itemBlocksize)  # zero-based

            keys = zip(*map(tuple, unravel_index(linearIdx, dims, order='F')))
            return zip(keys, buf)

        # map over blocks
        return (self.sc.parallelize(range(0, nblocks), nblocks).flatMap(lambda bn: readBlock(bn)),
                len(filenames), newDtype)

    @staticmethod
    def __readMetadataFromFirstPageOfMultiTif(reader, filePath):
        import thunder.rdds.fileio.multitif as multitif

        # read first page of first file to get expected image size
        tiffFP = reader.open(filePath)
        tiffParser = multitif.TiffParser(tiffFP, debug=False)
        tiffHeaders = multitif.TiffData()
        tiffParser.parseFileHeader(destinationTiff=tiffHeaders)
        firstIfd = tiffParser.parseNextImageFileDirectory(destinationTiff=tiffHeaders)
        if not firstIfd.isLuminanceImage():
            raise ValueError(("File %s does not appear to be a luminance " % filePath) +
                             "(greyscale or bilevel) TIF image, " +
                             "which are the only types currently supported")

        # keep reading pages until we reach the end of the file, in order to get number of planes:
        while tiffParser.parseNextImageFileDirectory(destinationTiff=tiffHeaders):
            pass

        # get dimensions
        npages = len(tiffHeaders.ifds)
        height = firstIfd.getImageHeight()
        width = firstIfd.getImageWidth()

        # get datatype
        bitsPerSample = firstIfd.getBitsPerSample()
        if not (bitsPerSample in (8, 16, 32, 64)):
            raise ValueError("Only 8, 16, 32, or 64 bit per pixel TIF images are supported, got %d" % bitsPerSample)

        sampleFormat = firstIfd.getSampleFormat()
        if sampleFormat == multitif.SAMPLE_FORMAT_UINT:
            dtStr = 'uint'
        elif sampleFormat == multitif.SAMPLE_FORMAT_INT:
            dtStr = 'int'
        elif sampleFormat == multitif.SAMPLE_FORMAT_FLOAT:
            dtStr = 'float'
        else:
            raise ValueError("Unknown TIF SampleFormat tag value %d, should be 1, 2, or 3 for uint, int, or float"
                             % sampleFormat)
        dtype = dtStr+str(bitsPerSample)

        return height, width, npages, dtype

    def _getSeriesBlocksFromMultiTif(self, dataPath, ext="tif", blockSize="150M",
                                     newDtype='smallfloat', casting='safe', startIdx=None, stopIdx=None,
                                     recursive=False):
        import thunder.rdds.fileio.multitif as multitif
        import itertools
        from PIL import Image
        import io

        dataPath = self.__normalizeDatafilePattern(dataPath, ext)
        blockSize = parseMemoryString(blockSize)

        reader = getFileReaderForPath(dataPath)(awsCredentialsOverride=self.awsCredentialsOverride)
        filenames = reader.list(dataPath, startIdx=startIdx, stopIdx=stopIdx, recursive=recursive)
        if not filenames:
            raise IOError("No files found for path '%s'" % dataPath)
        ntimepoints = len(filenames)

        doMinimizeReads = dataPath.lower().startswith("s3")
        # check PIL version to see whether it is actually pillow or indeed old PIL and choose
        # conversion function appropriately. See ImagesLoader.fromMultipageTif and common.pil_to_array
        # for more explanation.
        isPillow = hasattr(Image, "PILLOW_VERSION")
        if isPillow:
            conversionFcn = array  # use numpy's array() function
        else:
            from thunder.utils.common import pil_to_array
            conversionFcn = pil_to_array  # use our modified version of matplotlib's pil_to_array

        height, width, npages, dtype = SeriesLoader.__readMetadataFromFirstPageOfMultiTif(reader, filenames[0])
        if dtype.startswith('int'):
            raise ValueError('Signed integer tiff images are not supported in SeriesLoader (shuffle=False);' +
                             ' please try loading as Images (shuffle=True)')
        pixelBytesize = dtypeFunc(dtype).itemsize
        if newDtype is None or str(newDtype) == '':
            newDtype = str(dtype)
        elif newDtype == 'smallfloat':
            newDtype = str(smallestFloatType(dtype))
        else:
            newDtype = str(newDtype)

        # intialize at one block per plane
        bytesPerPlane = height * width * pixelBytesize * ntimepoints
        bytesPerBlock = bytesPerPlane
        blocksPerPlane = 1
        # keep dividing while cutting our size in half still leaves us bigger than the requested size
        # should end up no more than 2x blockSize.
        while bytesPerBlock >= blockSize * 2:
            bytesPerBlock /= 2
            blocksPerPlane *= 2

        blocklenPixels = max((height * width) / blocksPerPlane, 1)  # integer division
        while blocksPerPlane * blocklenPixels < height * width:  # make sure we're reading the plane fully
            blocksPerPlane += 1

        awsCredentialsOverride = self.awsCredentialsOverride

        # keys will be planeidx, blockidx:
        keys = list(itertools.product(xrange(npages), xrange(blocksPerPlane)))

        def readBlockFromTiff(planeIdxBlockIdx):
            planeIdx, blockIdx = planeIdxBlockIdx
            blocks = []
            planeShape = None
            blockStart = None
            blockEnd = None
            for fname in filenames:
                reader_ = getFileReaderForPath(fname)(awsCredentialsOverride=awsCredentialsOverride)
                fp = reader_.open(fname)
                try:
                    if doMinimizeReads:
                        # use multitif module to generate a fake, in-memory one-page tif file
                        # the advantage of this is that it cuts way down on the many small reads
                        # that PIL/pillow will make otherwise, which would be a problem for s3
                        tiffParser_ = multitif.TiffParser(fp, debug=False)
                        tiffFilebuffer = multitif.packSinglePage(tiffParser_, pageIdx=planeIdx)
                        byteBuf = io.BytesIO(tiffFilebuffer)
                        try:
                            pilImg = Image.open(byteBuf)
                            ary = conversionFcn(pilImg).T
                        finally:
                            byteBuf.close()
                        del tiffFilebuffer, tiffParser_, pilImg, byteBuf
                    else:
                        # read tif using PIL directly
                        pilImg = Image.open(fp)
                        pilImg.seek(planeIdx)
                        ary = conversionFcn(pilImg).T
                        del pilImg

                    if not planeShape:
                        planeShape = ary.shape[:]
                        blockStart = blockIdx * blocklenPixels
                        blockEnd = min(blockStart+blocklenPixels, planeShape[0]*planeShape[1])
                    blocks.append(ary.ravel(order='C')[blockStart:blockEnd])
                    del ary
                finally:
                    fp.close()

            buf = vstack(blocks).T  # dimensions are now linindex x time (images)
            del blocks
            buf = buf.astype(newDtype, casting=casting, copy=False)

            # append subscript keys based on dimensions
            linearIdx = arange(blockStart, blockEnd)  # zero-based

            seriesKeys = zip(*map(tuple, unravel_index(linearIdx, planeShape, order='C')))
            # add plane index to end of keys
            if npages > 1:
                seriesKeys = [tuple(list(keys_)[::-1]+[planeIdx]) for keys_ in seriesKeys]
            else:
                seriesKeys = [tuple(list(keys_)[::-1]) for keys_ in seriesKeys]
            return zip(seriesKeys, buf)

        # map over blocks
        rdd = self.sc.parallelize(keys, len(keys)).flatMap(readBlockFromTiff)
        if npages > 1:
            dims = (npages, width, height)
        else:
            dims = (width, height)

        metadata = (dims, ntimepoints, newDtype)
        return rdd, metadata

    def fromStack(self, dataPath, dims, ext="stack", blockSize="150M", dtype='int16',
                  newDtype='smallfloat', casting='safe', startIdx=None, stopIdx=None, recursive=False):
        """Load a Series object directly from binary image stack files.

        Parameters
        ----------

        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        dtype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newDtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startIdx, stopIdx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `dataPath` and `ext`. Interpreted according to python slice indexing conventions.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).
        """
        seriesBlocks, npointsInSeries, newDtype = \
            self._getSeriesBlocksFromStack(dataPath, dims, ext=ext, blockSize=blockSize, dtype=dtype,
                                           newDtype=newDtype, casting=casting, startIdx=startIdx, stopIdx=stopIdx,
                                           recursive=recursive)
        return Series(seriesBlocks, dims=dims, dtype=newDtype, index=arange(npointsInSeries),
                      awsCredentials=self.awsCredentialsOverride)

    def fromTif(self, dataPath, ext="tif", blockSize="150M", newDtype='smallfloat', casting='safe',
                startIdx=None, stopIdx=None, recursive=False):
        """Load a Series object from multipage tiff files.

        Parameters
        ----------

        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

        ext: string, optional, default "tif"
            Extension required on data files to be loaded.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        newDtype: dtype or dtype specifier or string 'smallfloat' or None, optional, default 'smallfloat'
            Numpy dtype of output series data. Most methods expect Series data to be floating-point. Input data will be
            cast to the requested `newdtype` if not None - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startIdx, stopIdx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `dataPath` and `ext`. Interpreted according to python slice indexing conventions.

        recursive: boolean, default False
            If true, will recursively descend directories rooted at dataPath, loading all files in the tree that
            have an extension matching 'ext'. Recursive loading is currently only implemented for local filesystems
            (not s3).
        """
        seriesBlocks, metadata = self._getSeriesBlocksFromMultiTif(dataPath, ext=ext, blockSize=blockSize,
                                                                   newDtype=newDtype, casting=casting,
                                                                   startIdx=startIdx, stopIdx=stopIdx,
                                                                   recursive=recursive)
        dims, npointsInSeries, dtype = metadata
        return Series(seriesBlocks, dims=Dimensions.fromTuple(dims[::-1]), dtype=dtype,
                      index=arange(npointsInSeries), awsCredentials=self.awsCredentialsOverride)

    def __saveSeriesRdd(self, seriesBlocks, outputDirPath, dims, npointsInSeries, dtype, overwrite=False):
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self.awsCredentialsOverride)
            overwrite = True  # prevent additional downstream checks for this path
        writer = getParallelWriterForPath(outputDirPath)(outputDirPath, overwrite=overwrite,
                                                         awsCredentialsOverride=self.awsCredentialsOverride)

        def blockToBinarySeries(kvIter):
            label = None
            keyPacker = None
            buf = StringIO()
            for seriesKey, series in kvIter:
                if keyPacker is None:
                    keyPacker = struct.Struct('h'*len(seriesKey))
                    label = SimpleBlocks.getBinarySeriesNameForKey(seriesKey) + ".bin"
                buf.write(keyPacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return [(label, val)]

        seriesBlocks.mapPartitions(blockToBinarySeries).foreach(writer.writerFcn)
        writeSeriesConfig(outputDirPath, len(dims), npointsInSeries, valueType=dtype, overwrite=overwrite,
                          awsCredentialsOverride=self.awsCredentialsOverride)

    def saveFromStack(self, dataPath, outputDirPath, dims, ext="stack", blockSize="150M", dtype='int16',
                      newDtype=None, casting='safe', startIdx=None, stopIdx=None, overwrite=False, recursive=False):
        """Write out data from binary image stack files in the Series data flat binary format.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

        outputDirPath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in dataPath.

        dims: tuple of positive int
            Dimensions of input image data, ordered with the fastest-changing dimension first.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        dtype: dtype or dtype specifier, optional, default 'int16'
            Numpy dtype of input stack data

        newDtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default None
            Numpy dtype of output series binary data. Input data will be cast to the requested `newdtype` if not None
            - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startIdx, stopIdx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `dataPath` and `ext`. Interpreted according to python slice indexing conventions.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputdirpath will first be deleted, along with all its contents, if it
            already exists. If false, a ValueError will be thrown if outputdirpath is found to already exist.

        """
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self.awsCredentialsOverride)
            overwrite = True  # prevent additional downstream checks for this path

        seriesBlocks, npointsInSeries, newDtype = \
            self._getSeriesBlocksFromStack(dataPath, dims, ext=ext, blockSize=blockSize, dtype=dtype,
                                           newDtype=newDtype, casting=casting, startIdx=startIdx, stopIdx=stopIdx,
                                           recursive=recursive)

        self.__saveSeriesRdd(seriesBlocks, outputDirPath, dims, npointsInSeries, newDtype, overwrite=overwrite)

    def saveFromTif(self, dataPath, outputDirPath, ext="tif", blockSize="150M",
                    newDtype=None, casting='safe', startIdx=None, stopIdx=None,
                    overwrite=False, recursive=False):
        """Write out data from multipage tif files in the Series data flat binary format.

        Parameters
        ----------
        dataPath: string
            Path to data files or directory, specified as either a local filesystem path or in a URI-like format,
            including scheme. A dataPath argument may include a single '*' wildcard character in the filename.

        outputDirPpath: string
            Path to a directory into which to write Series file output. An outputdir argument may be either a path
            on the local file system or a URI-like format, as in dataPath.

        ext: string, optional, default "stack"
            Extension required on data files to be loaded.

        blockSize: string formatted as e.g. "64M", "512k", "2G", or positive int. optional, default "150M"
            Requested size of Series partitions in bytes (or kilobytes, megabytes, gigabytes).

        newDtype: floating-point dtype or dtype specifier or string 'smallfloat' or None, optional, default None
            Numpy dtype of output series binary data. Input data will be cast to the requested `newdtype` if not None
            - see Data `astype()` method.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's `astype()` method; see numpy documentation for details.

        startIdx, stopIdx: nonnegative int. optional.
            Indices of the first and last-plus-one data file to load, relative to the sorted filenames matching
            `dataPath` and `ext`. Interpreted according to python slice indexing conventions.

        overwrite: boolean, optional, default False
            If true, the directory specified by outputdirpath will first be deleted, along with all its contents, if it
            already exists. If false, a ValueError will be thrown if outputdirpath is found to already exist.

        """
        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputDirPath, self.awsCredentialsOverride)
            overwrite = True  # prevent additional downstream checks for this path

        seriesBlocks, metadata = self._getSeriesBlocksFromMultiTif(dataPath, ext=ext, blockSize=blockSize,
                                                                   newDtype=newDtype, casting=casting,
                                                                   startIdx=startIdx, stopIdx=stopIdx,
                                                                   recursive=recursive)
        dims, npointsInSeries, dtype = metadata
        self.__saveSeriesRdd(seriesBlocks, outputDirPath, dims, npointsInSeries, dtype, overwrite=overwrite)

    def fromMatLocal(self, dataPath, varName, keyFile=None):
        """Loads Series data stored in a Matlab .mat file.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = loadmat(dataPath)[varName]
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyFile:
            keys = map(lambda x: tuple(x), loadmat(keyFile)['keys'])
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype),
                     awsCredentials=self.awsCredentialsOverride)

        return rdd

    def fromNpyLocal(self, dataPath, keyFile=None):
        """Loads Series data stored in the numpy save() .npy format.

        `datafile` must refer to a path visible to all workers, such as on NFS or similar mounted shared filesystem.
        """
        data = load(dataPath)
        if data.ndim > 2:
            raise IOError('Input data must be one or two dimensional')
        if keyFile:
            keys = map(lambda x: tuple(x), load(keyFile))
        else:
            keys = arange(0, data.shape[0])

        rdd = Series(self.sc.parallelize(zip(keys, data), self.minPartitions), dtype=str(data.dtype),
                     awsCredentials=self.awsCredentialsOverride)

        return rdd

    def loadConf(self, dataPath, confFilename='conf.json'):
        """Returns a dict loaded from a json file.

        Looks for file named `conffile` in same directory as `dataPath`

        Returns {} if file not found
        """
        if not confFilename:
            return {}

        reader = getFileReaderForPath(dataPath)(awsCredentialsOverride=self.awsCredentialsOverride)
        try:
            jsonBuf = reader.read(dataPath, filename=confFilename)
        except FileNotFoundError:
            return {}

        params = json.loads(jsonBuf)

        if 'format' in params:
            raise Exception("Numerical format of value should be specified as 'valuetype', not 'format'")
        if 'keyformat' in params:
            raise Exception("Numerical format of key should be specified as 'keytype', not 'keyformat'")

        return params


def writeSeriesConfig(outputDirPath, nkeys, nvalues, keyType='int16', valueType='int16',
                      confFilename="conf.json", overwrite=True, awsCredentialsOverride=None):
    """Helper function to write out a conf.json file with required information to load Series binary data.
    """
    import json
    from thunder.rdds.fileio.writers import getFileWriterForPath

    filewriterClass = getFileWriterForPath(outputDirPath)
    # write configuration file
    # config JSON keys are lowercased "valuetype", "keytype", not valueType, keyType
    conf = {'input': outputDirPath,
            'nkeys': nkeys, 'nvalues': nvalues,
            'valuetype': str(valueType), 'keytype': str(keyType)}

    confWriter = filewriterClass(outputDirPath, confFilename, overwrite=overwrite,
                                 awsCredentialsOverride=awsCredentialsOverride)
    confWriter.writeFile(json.dumps(conf, indent=2))

    # touch "SUCCESS" file as final action
    successWriter = filewriterClass(outputDirPath, "SUCCESS", overwrite=overwrite,
                                    awsCredentialsOverride=awsCredentialsOverride)
    successWriter.writeFile('')
