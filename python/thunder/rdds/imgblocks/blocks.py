"""Classes relating to Blocks, which represent subdivided Images data.
"""
import cStringIO as StringIO
import itertools
from numpy import zeros, arange
import struct
from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions


def getStartStopStep(slise, refSize):
    """Extracts the start, stop, and step parameters from a passed slice.

    If the slice is slice(None), returns refsize as the stop position. start will be 0 and
    step 1 in this case, which corresponds to normal python slice semantics.
    """
    stop = refSize if slise.stop is None else slise.stop
    start = 0 if slise.start is None else slise.start
    step = 1 if slise.step is None else slise.step
    return start, stop, step


def sliceToXRange(slise, stop):
    return xrange(*getStartStopStep(slise, stop))


def slicesToIterators(slices, shape):
    """Returns a sequence of iterators over the range of the passed slices.

    The output of this function is expected to be passed into itertools.product.

    Parameters
    ----------
    slices: sequence of slices

    shape: tuple of positive int, same length as slices

    Returns
    -------
    sequence of iterators, one iterator per slice
    """
    return [sliceToXRange(slise, size) for slise, size in zip(slices, shape)]


class Blocks(Data):
    """Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Data._metadata + ['_dims', '_nimages']

    def __init__(self, rdd, dims=None, nimages=None, dtype=None, awsCredentials=None):
        super(Blocks, self).__init__(rdd, dtype=dtype, awsCredentials=awsCredentials)
        self._dims = dims
        self._nimages = nimages

    @property
    def dims(self):
        """Shape of the original Images data from which these Blocks were derived.

        n-tuple of positive int
        """
        if not self._dims:
            self.populateParamsFromFirstRecord()
        return self._dims

    @property
    def nimages(self):
        """Number of images (records) in the original Images data from which these Blocks were derived.

        positive int
        """
        return self._nimages

    def _resetCounts(self):
        pass

    def toSeries(self):
        """Returns a Series Data object.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError("toSeries not implemented")

    def toBinarySeries(self):
        """Returns an RDD of binary series data.

        The keys of a binary series RDD should be filenames ending in ".bin".
        The values should be packed binary data.

        Subclasses that can be converted to a Series object are expected to override this method.
        """
        raise NotImplementedError("toBinarySeries not implemented")

    def saveAsBinarySeries(self, outputDirPath, overwrite=False):
        """Writes out Series-formatted data.

        Subclasses are *not* expected to override this method.

        Parameters
        ----------
        outputdirname : string path or URI to directory to be created
            Output files will be written underneath outputdirname. This directory must not yet exist
            (unless overwrite is True), and must be no more than one level beneath an existing directory.
            It will be created as a result of this call.

        overwrite : bool
            If true, outputdirname and all its contents will be deleted and recreated as part
            of this call.
        """
        from thunder.rdds.fileio.writers import getParallelWriterForPath
        from thunder.rdds.fileio.seriesloader import writeSeriesConfig

        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=self._awsCredentials)
            overwrite = True  # prevent additional downstream checks for this path

        writer = getParallelWriterForPath(outputDirPath)(outputDirPath, overwrite=overwrite,
                                                         awsCredentialsOverride=self._awsCredentials)

        binseriesRdd = self.toBinarySeries()

        binseriesRdd.foreach(writer.writerFcn)
        writeSeriesConfig(outputDirPath, len(self.dims), self.nimages, keyType='int16', valueType=self.dtype,
                          overwrite=overwrite, awsCredentialsOverride=self._awsCredentials)


class SimpleBlocks(Blocks):
    """Basic concrete implementation of Blocks.

    These Blocks will be contiguous, nonoverlapping subsections of the original Images arrays.
    """
    @property
    def _constructor(self):
        return SimpleBlocks

    def populateParamsFromFirstRecord(self):
        record = super(SimpleBlocks, self).populateParamsFromFirstRecord()
        self._dims = Dimensions.fromTuple(record[0].origShape)
        return record

    @staticmethod
    def _toSeriesIter(blockGroupingKey, blockArrayValue):
        """Generator yielding an iteration over (spatial key, array) pairs suitable for casting into a Series object.

        Parameters:
        -----------
        blockGroupingKey: BlockGroupingKey instance associated with the passed array value

        blockArrayValue: numpy array
            Array data for the block associated with the passed key

        Yields:
        --------
        iterator< key, series >
        key: tuple of int
        series: 1d array of self.values.dtype
        """
        # staticmethod declaration appears necessary here to avoid python anonymous function / closure /
        # serialization weirdness
        for spatialIndices in blockGroupingKey.spatialIndexRange():
            series = blockGroupingKey.getSeriesDataForSpatialIndices(spatialIndices, blockArrayValue)
            yield spatialIndices, series

    @staticmethod
    def _toTimeSlicedBlocksIter(blockGroupingKey, blockArrayValue):
        """Generator function that yields an iteration over (reconstructionKey, numpy array)
        pairs.

        Parameters:
        -----------
        blockGroupingKey: BlockGroupingKey instance associated with the passed array value

        blockArrayValue: numpy array
            Array data for the block associated with the passed key

        Yields:
        -------
        iterator <ImageReconstructionKey, numpy array>
        ImageReconstructionKey: new key
            Key imgSlices, origShape will be for full Image space, not just block, and will not include time
        array: numpy array with dimensions x, y, z equal to block shape; no time dimension
        """
        # set up new slices:
        newImgSlices = blockGroupingKey.imgSlices[1:]
        newOrigShape = blockGroupingKey.origShape[1:]
        for tpIdx in blockGroupingKey.temporalIndexRange():
            # new array value:
            newVal = blockGroupingKey.getImageDataForTemporalIndex(tpIdx, blockArrayValue)
            newKey = ImageReconstructionKey(tpIdx, newOrigShape, newImgSlices)
            yield newKey, newVal

    @staticmethod
    def _combineTimeSlicedBlocks(temporalIdxAndSlicedSequence):
        temporalIdx, slicedSequence = temporalIdxAndSlicedSequence
        # sequence will be of (image reconstruction key, numpy array) pairs
        ary = None
        for key, block in slicedSequence:
            if ary is None:
                # set up collection array:
                ary = zeros(key.origShape, block.dtype)
            # put values into collection array:
            ary[key.imgSlices] = block

        return temporalIdx, ary

    def toSeries(self, newDType="smallfloat", casting="safe"):
        from thunder.rdds.series import Series
        # returns generator of (z, y, x) array data for all z, y, x
        seriesRdd = self.rdd.flatMap(lambda kv: SimpleBlocks._toSeriesIter(kv[0], kv[1]))

        idx = arange(self._nimages) if self._nimages else None
        return Series(seriesRdd, dims=self.dims, index=idx, dtype=self.dtype, awsCredentials=self._awsCredentials)\
            .astype(newDType, casting=casting)

    def toImages(self):
        from thunder.rdds.images import Images
        timeRdd = self.rdd.flatMap(lambda kv: SimpleBlocks._toTimeSlicedBlocksIter(kv[0], kv[1]))
        timeSortedRdd = timeRdd.groupBy(lambda (k, _): k.temporalKey).sortByKey()
        imagesRdd = timeSortedRdd.map(SimpleBlocks._combineTimeSlicedBlocks)
        return Images(imagesRdd, dims=self._dims, nrecords=self._nimages, dtype=self._dtype,
                      awsCredentials=self._awsCredentials)

    @staticmethod
    def getBinarySeriesNameForKey(blockKey):
        """

        Returns
        -------
        string blockLabel
            Block label will be in form "key02_0000k-key01_0000j-key00_0000i" for i,j,k x,y,z indicies as first series
            in block. No extension (e.g. ".bin") is appended to the block label.
        """
        return '-'.join(reversed(["key%02d_%05g" % (ki, k) for (ki, k) in enumerate(blockKey)]))

    def toBinarySeries(self):

        def blockToBinarySeries(kv):
            blockKey, blockVal = kv
            label = SimpleBlocks.getBinarySeriesNameForKey(blockKey.spatialKey)+".bin"
            keyPacker = None
            buf = StringIO.StringIO()
            for seriesKey, series in SimpleBlocks._toSeriesIter(blockKey, blockVal):
                if keyPacker is None:
                    keyPacker = struct.Struct('h'*len(seriesKey))
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keyPacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            return label, val

        return self.rdd.map(blockToBinarySeries)


class PaddedBlocks(SimpleBlocks):

    @property
    def _constructor(self):
        return PaddedBlocks


class BlockingKey(object):
    @property
    def temporalKey(self):
        raise NotImplementedError

    @property
    def spatialKey(self):
        raise NotImplementedError


class ImageReconstructionKey(BlockingKey):
    def __init__(self, timeIdx, origShape, imgSlices):
        self.timeIdx = timeIdx
        self.origShape = origShape
        self.imgSlices = imgSlices

    @property
    def temporalKey(self):
        return self.timeIdx

    @property
    def spatialKey(self):
        return tuple(self.imgSlices)

    def __repr__(self):
        return "ImageReconstructionKey\ntimeIdx: %d\norigShape: %s\nimgSlices: %s" % \
               (self.timeIdx, self.origShape, self.imgSlices)


class BlockGroupingKey(BlockingKey):
    """Key used to extract and sort SimpleBlocks.

    These keys are expected to be used in a Spark key-value RDD, where the value is a numpy array representing
    a contiguous subsection of a 4d spatiotemporal image.

    Attributes
    ----------
    origShape: sequence of positive int
        Shape of original Images array of which this block is a part. This shape includes a "time" dimension as the
        first dimension. (This additional dimension is not present on Images values, which are each assumed to represent
        a single point in time.)

    imgSlices: sequence of slices
        Slices into an array of shape origShape; these slices represent the full extent of the block in its original
        space. These slices include the temporal dimension as the first slice in the sequence (imgSlices[0]).
    """
    def __init__(self, origShape, imgSlices):
        self.origShape = origShape
        self.imgSlices = imgSlices

    @property
    def temporalKey(self):
        # temporal key is index of time point, obtained from first slice (time dimension)
        start = self.imgSlices[0].start
        return start if not (start is None) else 0

    @property
    def spatialKey(self):
        return tuple(sl.start for sl in self.imgSlices[1:])

    def asTemporallyConcatenatedKey(self):
        """Returns a new key that is a copy of self, except that the new temporal range of imgSlices will be from 0 to
        the total number of time points.

        Used in SimpleBlockingStrategy.combiningFcn to generate new key for a concatenated set of spatially-aligned
        blocks grouped together across time.
        """
        # new slices should be full slice for formerly planar dimension, plus existing block slices
        newImgSlices = [slice(0, self.origShape[0])] + list(self.imgSlices)[1:]
        return BlockGroupingKey(origShape=self.origShape, imgSlices=tuple(newImgSlices))

    def getSeriesDataForSpatialIndices(self, spatialIndices, ary):
        """Returns a one-dimensional array corresponding to the time series (dimension 0) at the passed spatial
        indices, given in original image coordinates

        Parameters
        ----------
        spatialIndices: sequence of int of length == ary.ndim - 1
            spatial coordinates in original image space, ordered as x, y, z. Time dimension is not included.
        """
        slices = [slice(None)]  # start with full slice in temporal dimension, dim 0
        for idx, origSlice in zip(spatialIndices, self.imgSlices[1:]):
            # transform index from original image space into block array space
            # requires any offset in image space to be subtracted from our starting position
            start = idx - origSlice.start if not (origSlice.start is None) else idx
            newSlice = slice(start, start+1, 1)
            slices.append(newSlice)
        return ary[slices].squeeze()

    def getImageDataForTemporalIndex(self, temporalIndex, ary):
        return ary[temporalIndex]

    def spatialIndexRange(self):
        """Generator function yielding spatial indices as (x, y, z) index tuples.

        The indices yielded by this method will cover the full spatial extent of the block with which this key is
        associated.

        The indices will be ordered so that the last index (z-dimension) will be incrementing most quickly. This should
        make these indices suitable to be cast directly into keys for Series data objects.

        The indices yielded here can be passed into self.getSeriesDataForSpatialIndices() to get the associated 1d time
        series array.

        Yields
        ------
        spatial indices, tuple of nonnegative int
        """
        rangeIters = slicesToIterators(self.imgSlices[1:], self.origShape[1:])
        for idxSeq in itertools.product(*reversed(rangeIters)):
            yield tuple(reversed(idxSeq))

    def temporalIndexRange(self):
        """Returns an iterable object over the range of time points represented by this key.

        Returns
        -------
        iterable over nonnegative int
        """
        return sliceToXRange(self.imgSlices[0], self.origShape[0])

    def __repr__(self):
        return "BlockGroupingKey\norigShape: %s\nimgSlices: %s" % (self.origShape, self.imgSlices)


class PaddedBlockGroupingKey(BlockGroupingKey):
    """Key used to extract and sort PaddedBlocks.

    These keys are expected to be used in a Spark key-value RDD, where the value is a numpy array representing
    a contiguous subsection of a 4d spatiotemporal image.

    Attributes
    ----------
    origShape: sequence of positive int
        Shape of original Images array of which this block is a part. This shape includes a "time" dimension as the
        first dimension. (This additional dimension is not present on Images values, which are each assumed to represent
        a single point in time.)

    padImgSlices: sequence of slices
        Slices into an array of shape origShape; these slices represent the full extent of the block in its original
        space, including padding. These slices include the temporal dimension as the first slice in the sequence
        (padImgSlices[0])

    imgSlices: sequence of slices
        Slices into an array of shape origShape; these slices represent the 'core' block, without padding, in its
        original space. These slices include the temporal dimension as the first slice in the sequence
        (imgSlices[0])

    valShape: tuple of positive int
        Shape of associated numpy array value

    valSlices: sequence of slices
        Slices into the array-type value for which this object is the key in a Spark key-value RDD. These slices
        represent the 'core' block, without padding. These slices should be the same size as those in imgSlices,
        differing only in their offsets.

    """
    def __init__(self, origShape, padImgSlices, imgSlices, valShape, valSlices):
        super(PaddedBlockGroupingKey, self).__init__(origShape, imgSlices)
        self.padImgSlices = padImgSlices
        self.valShape = valShape
        self.valSlices = valSlices

    def asTemporallyConcatenatedKey(self):
        """Returns a new key that is a copy of self, except that the new temporal range is from 0 to the total number
        of time points.
        """
        allTimeSlice = slice(0, self.origShape[0])
        newImgSlices = [allTimeSlice] + list(self.imgSlices[1:])
        newPadImgSlices = [allTimeSlice] + list(self.padImgSlices[1:])
        newValShape = [self.origShape[0]] + list(self.valShape[1:])
        newvalSlices = [allTimeSlice] + list(self.valSlices[1:])
        return PaddedBlockGroupingKey(origShape=self.origShape, imgSlices=tuple(newImgSlices),
                                      padImgSlices=tuple(newPadImgSlices), valShape=tuple(newValShape),
                                      valSlices=tuple(newvalSlices))

    def getSeriesDataForSpatialIndices(self, spatialIndices, ary):
        """Returns a one-dimensional array corresponding to the time series (dimension 0) at the passed spatial
        indices, given in original image coordinates

        This implementation overrides that in BlockGroupingKey, taking into account padding on the block side
        as described in self.valSlices.

        Parameters
        ----------
        spatialIndices: sequence of int of length == ary.ndim - 1
            spatial coordinates in original image space, ordered as x, y, z
        """
        slices = [slice(None)]  # start with full slice in temporal dimension, dim 0
        for spatialIndex, origSlice, valSlice, valSize, imgSize in zip(spatialIndices, self.imgSlices[1:],
                                                                       self.valSlices[1:], self.valShape[1:],
                                                                       self.origShape[1:]):
            # transform index from original image space into block array space
            # requires any offset in image space to be subtracted from our starting position

            valStart, valStop, valStep = \
                getStartStopStep(valSlice, valSize)
            imgStart, imgStop, imgStep = \
                getStartStopStep(origSlice, imgSize)
            stepNum = (spatialIndex - imgStart) / imgStep
            valIdx = valStart + stepNum * valStep
            slices.append(slice(valIdx, valIdx+1, 1))

        return ary[slices].squeeze()

    def getImageDataForTemporalIndex(self, tpIdx, ary):
        return ary[tpIdx][self.valSlices[1:]]

    def __repr__(self):
        return "PaddedBlockGroupingKey\norigShape: %s\npadImgSlices: %s\nimgSlices: %s\nvalShape: %s\nvalSlices: %s)" % \
               (self.origShape, self.padImgSlices, self.imgSlices, self.valShape, self.valSlices)