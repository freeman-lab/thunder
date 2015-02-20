from numpy import ndarray, array, sum, mean, median, std, size, arange, \
    percentile, asarray, maximum, zeros, corrcoef, where, \
    true_divide, ceil

from thunder.rdds.data import Data
from thunder.rdds.keys import Dimensions
from thunder.utils.common import loadMatVar


class Series(Data):
    """
    Distributed collection of 1d array data with axis labels.

    Backed by an RDD of key-value pairs, where the
    key is a tuple identifier, and the value is a one-dimensional array of floating-point values.
    It also has a fixed index to represent a label for each value in the arrays.
    Can optionally store and use the dimensions of the keys (min, max, and count).

    Series data will be automatically cast to a floating-point value on loading if its on-disk
    representation is integer valued.

    Parameters
    ----------

    rdd : RDD of (tuple, array) pairs
        RDD containing the series data

    index : array-like or one-dimensional list
        Values must be unique, same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    dims : Dimensions
        Specify the dimensions of the keys (min, max, and count), can
        avoid computation if known in advance

    See also
    --------
    TimeSeries : a Series where the indices represent time
    SpatialSeries : a Series where the keys represent spatial coordinates
    """

    _metadata = Data._metadata + ['_dims', '_index']

    def __init__(self, rdd, nrecords=None, dtype=None, awsCredentials=None, index=None, dims=None):
        super(Series, self).__init__(rdd, nrecords=nrecords, dtype=dtype, awsCredentials=awsCredentials)
        self._index = None
        if index is not None:
            self.index = index
        if dims and not isinstance(dims, Dimensions):
            try:
                dims = Dimensions.fromTuple(dims)
            except:
                raise TypeError("Series dims parameter must be castable to Dimensions object, got: %s" % str(dims))
        self._dims = dims

    @property
    def index(self):
        if self._index is None:
            self.populateParamsFromFirstRecord()
        return self._index
        
    @index.setter
    def index(self, value):
        # touches self.index to trigger automatic calculation from first record if self.index is not set
        lenSelf = len(self.index)
        if type(value) is str:
            value = [value]
        # if new index is not indexable, assume that it is meant as an index of length 1
        try:
            value[0]
        except:
            value = [value]
        try:
            lenValue = len(value)
        except:
            raise TypeError("Index must be an object with a length")
        if lenValue != lenSelf:
            raise ValueError("Length of new index ({0}) must match length of original index ({1})".format(lenValue, lenSelf))
        self._index = value

    @property
    def dims(self):
        from thunder.rdds.keys import Dimensions
        if self._dims is None:
            entry = self.populateParamsFromFirstRecord()[0]
            n = size(entry)
            d = self.rdd.keys().mapPartitions(lambda i: [Dimensions(i, n)]).reduce(lambda x, y: x.mergeDims(y))
            self._dims = d
        return self._dims

    @property
    def dtype(self):
        # override just calls superclass; here for explicitness
        return super(Series, self).dtype

    def populateParamsFromFirstRecord(self):
        """Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        Returns the result of calling self.rdd.first().
        """
        record = super(Series, self).populateParamsFromFirstRecord()
        if self._index is None:
            val = record[1]
            try:
                l = len(val)
            except TypeError:
                # TypeError thrown after calling len() on object with no __len__ method
                l = 1
            self._index = arange(0, l)
        return record

    @property
    def _constructor(self):
        return Series

    @staticmethod
    def _checkType(record):
        key = record[0]
        value = record[1]
        if not isinstance(key, tuple):
            raise Exception('Keys must be tuples')
        if not isinstance(value, ndarray):
            raise Exception('Values must be ndarrays')
        else:
            if value.ndim != 1:
                raise Exception('Values must be 1d arrays')

    def _resetCounts(self):
        self._dims = None
        return self

    def between(self, left, right, inclusive=True):
        """
        Select subset of values within the given index range

        Parameters
        ----------
        left : int
            Left-most index in the desired range

        right: int
            Right-most index in the desired range

        inclusive : boolean, optional, default = True
            Whether selection should include bounds
        """
        if inclusive:
            crit = lambda x: left <= x <= right
        else:
            crit = lambda x: left < x < right
        return self.select(crit)

    def select(self, crit):
        """
        Select subset of values that match a given index criterion

        Parameters
        ----------
        crit : function, list, str, int
            Criterion function to apply to indices, specific index value,
            or list of indices
        """

        import types

        # handle lists, strings, and ints
        if not isinstance(crit, types.FunctionType):
            # set("foo") -> {"f", "o"}; wrap in list to prevent:
            if isinstance(crit, basestring):
                critlist = set([crit])
            else:
                try:
                    critlist = set(crit)
                except TypeError:
                    # typically means crit is not an iterable type; for instance, crit is an int
                    critlist = set([crit])
            crit = lambda x: x in critlist

        # if only one index, return it directly or throw an error
        index = self.index
        if size(index) == 1:
            if crit(index):
                return self
            else:
                raise Exception("No indices found matching criterion")

        # determine new index and check the result
        newIndex = [i for i in index if crit(i)]
        if len(newIndex) == 0:
            raise Exception("No indices found matching criterion")
        if array(newIndex == index).all():
            return self

        # use fast logical indexing to get the new values
        subInds = where(map(lambda x: crit(x), index))
        rdd = self.rdd.mapValues(lambda x: x[subInds])

        # if singleton, need to check whether it's an array or a scalar/int
        # if array, recompute a new set of indices
        if len(newIndex) == 1:
            rdd = rdd.mapValues(lambda x: x[0])
            val = rdd.first()[1]
            if size(val) == 1:
                newIndex = newIndex[0]
            else:
                newIndex = arange(0, size(val))

        return self._constructor(rdd, index=newIndex).__finalize__(self)

    def center(self, axis=0):
        """ Center series data by subtracting the mean
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.applyValues(lambda x: x - mean(x))
        elif axis == 1:
            meanVec = self.mean()
            return self.applyValues(lambda x: x - meanVec)
        else:
            raise Exception('Axis must be 0 or 1')

    def standardize(self, axis=0):
        """ Standardize series data by dividing by the standard deviation
        either within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to standardize along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.applyValues(lambda x: x / std(x))
        elif axis == 1:
            stdvec = self.stdev()
            return self.applyValues(lambda x: x / stdvec)
        else:
            raise Exception('Axis must be 0 or 1')

    def zscore(self, axis=0):
        """ Zscore series data by subtracting the mean and
        dividing by the standard deviation either
        within or across records

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, rows (0) or columns (1)
        """
        if axis == 0:
            return self.applyValues(lambda x: (x - mean(x)) / std(x))
        elif axis == 1:
            stats = self.stats()
            meanVec = stats.mean()
            stdVec = stats.stdev()
            return self.applyValues(lambda x: (x - meanVec) / stdVec)
        else:
            raise Exception('Axis must be 0 or 1')

    def correlate(self, signal, var='s'):
        """
        Correlate series data against one or many one-dimensional arrays

        Parameters
        ----------
        signal : array, or str
            Signal(s) to correlate against, can be a numpy array or a
            MAT file containing the signal as a variable

        var : str
            Variable name if loading from a MAT file
        """
        from scipy.io import loadmat

        if type(signal) is str:
            s = loadmat(signal)[var]
        else:
            s = asarray(signal)

        # handle the case of a 1d signal
        if s.ndim == 1:
            if size(s) != size(self.index):
                raise Exception('Size of signal to correlate with, %g, does not match size of series' % size(s))
            rdd = self.rdd.mapValues(lambda x: corrcoef(x, s)[0, 1])
            newIndex = 0
        # handle multiple 1d signals
        elif s.ndim == 2:
            if s.shape[1] != size(self.index):
                raise Exception('Length of signals to correlate with, %g, does not match size of series' % s.shape[1])
            rdd = self.rdd.mapValues(lambda x: array([corrcoef(x, y)[0, 1] for y in s]))
            newIndex = range(0, s.shape[0])
        else:
            raise Exception('Signal to correlate with must have 1 or 2 dimensions')

        # return result
        return self._constructor(rdd, dtype='float64', index=newIndex).__finalize__(self)

    def seriesMax(self):
        """ Compute the value maximum of each record in a Series """
        return self.seriesStat('max')

    def seriesMin(self):
        """ Compute the value minimum of each record in a Series """
        return self.seriesStat('min')

    def seriesSum(self):
        """ Compute the value sum of each record in a Series """
        return self.seriesStat('sum')

    def seriesMean(self):
        """ Compute the value mean of each record in a Series """
        return self.seriesStat('mean')

    def seriesMedian(self):
        """ Compute the value median of each record in a Series """
        return self.seriesStat('median')

    def seriesPercentile(self, q):
        """ Compute the value percentile of each record in a Series.
        
        Parameters

          q: a floating point number between 0 and 100 inclusive.
        """
        rdd = self.rdd.mapValues(lambda x: percentile(x, q))
        return self._constructor(rdd, index=q).__finalize__(self, noPropagate=('_dtype',))

    def seriesStdev(self):
        """ Compute the value std of each record in a Series """
        return self.seriesStat('stdev')

    def seriesStat(self, stat):
        """ Compute a simple statistic for each record in a Series

        Parameters
        ----------
        stat : str
            Which statistic to compute
        """
        STATS = {
            'sum': sum,
            'mean': mean,
            'median': median,
            'stdev': std,
            'max': max,
            'min': min,
            'count': size
        }
        func = STATS[stat.lower()]
        rdd = self.rdd.mapValues(lambda x: func(x))
        return self._constructor(rdd, index=stat).__finalize__(self, noPropagate=('_dtype',))

    def seriesStats(self):
        """
        Compute a collection of statistics for each record in a Series
        """
        rdd = self.rdd.mapValues(lambda x: array([x.size, mean(x), std(x), max(x), min(x)]))
        return self._constructor(rdd, index=['count', 'mean', 'std', 'max', 'min'])\
            .__finalize__(self, noPropagate=('_dtype',))

    def maxProject(self, axis=0):
        """
        Project along one of the keys
        """
        import copy
        dims = copy.copy(self.dims)
        nkeys = len(self.first()[0])
        if axis > nkeys - 1:
            raise IndexError('only %g keys, cannot compute maximum along axis %g' % (nkeys, axis))
        rdd = self.rdd.map(lambda (k, v): (tuple(array(k)[arange(0, nkeys) != axis]), v)).reduceByKey(maximum)
        dims.min = list(array(dims.min)[arange(0, nkeys) != axis])
        dims.max = list(array(dims.max)[arange(0, nkeys) != axis])
        return self._constructor(rdd, dims=dims).__finalize__(self)

    def subToInd(self, order='F', isOneBased=True):
        """
        Convert subscript index keys to linear index keys

        Parameters
        ----------
        order : str, 'C' or 'F', default = 'F'
            Specifies row-major or column-major array indexing. See numpy.ravel_multi_index.

        isOneBased : boolean, default = True
            True if subscript indices start at 1, False if they start at 0
        """
        from thunder.rdds.keys import _subToIndConverter

        # converter = _subtoind_converter(self.dims.max, order=order, onebased=onebased)
        converter = _subToIndConverter(self.dims.count, order=order, isOneBased=isOneBased)
        rdd = self.rdd.map(lambda (k, v): (converter(k), v))
        return self._constructor(rdd, index=self._index).__finalize__(self)

    def indToSub(self, order='F', isOneBased=True, dims=None):
        """
        Convert linear indexing to subscript indexing

        Parameters
        ----------
        dims : array-like, optional
            Maximum dimensions. If not provided, will use dims property.

        order : str, 'C' or 'F', default = 'F'
            Specifies row-major or column-major array indexing. See numpy.unravel_index.

        onebased : boolean, default = True
            True if generated subscript indices are to start at 1, False to start at 0
        """
        from thunder.rdds.keys import _indToSubConverter

        if dims is None:
            dims = self.dims.max

        converter = _indToSubConverter(dims, order=order, isOneBased=isOneBased)
        rdd = self.rdd.map(lambda (k, v): (converter(k), v))
        return self._constructor(rdd, index=self._index).__finalize__(self)

    def pack(self, selection=None, sorting=False, transpose=False, dtype=None, casting='safe'):
        """
        Pack a Series into a local array (e.g. for saving)

        This operation constructs a multidimensional numpy array from the values in this Series object,
        with indexing into the returned array as implied by the Series RDD keys. The returned numpy
        array will be local to the Spark driver; the data set should be filtered down to a reasonable
        size (such as by seriesMean(), select(), or the `selection` parameter) before attempting to
        pack() a large data set.

        Parameters
        ----------
        selection : function, list, str, or int, optional, default None
            Criterion for selecting a subset, list, or index value

        sorting : boolean, optional, default False
            Whether to sort the local array based on the keys. In most cases the returned array will
            already be ordered correctly, and so an explicit sorting=True is typically not necessary.

        transpose : boolean, optional, default False
            Transpose the spatial dimensions of the returned array.

        dtype: numpy dtype, dtype specifier, or string 'smallfloat'. optional, default None.
            If present, will cast the values to the requested dtype before collecting on the driver. See Data.astype()
            and numpy's astype() function for details.

        casting: casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method if dtype is given; see numpy documentation for details.

        Returns
        -------
        result: numpy array
            An array with dimensionality inferred from the RDD keys. Data in an individual Series
            value will be placed into this returned array by interpreting the Series keys as indicies
            into the returned array. The shape of the returned array will be (num time points x spatial shape).
            For instance, a series derived from 4 2d images, each 64 x 128, will have dims.count==(64, 128)
            and will pack into an array with shape (4, 64, 128). If transpose is true, the spatial dimensions
            will be reversed, so that in this example the shape of the returned array will be (4, 128, 64).
        """

        if selection:
            out = self.select(selection)
        else:
            out = self

        if not (dtype is None):
            out = out.astype(dtype, casting)

        if sorting is True:
            result = out.sortByKey().values().collect()
        else:
            result = out.rdd.values().collect()

        nout = size(result[0])

        # reshape into a dense array of shape (b, x, y, z)  or (b, x, y) or (b, x)
        # where b is the number of outputs per record
        out = asarray(result).reshape(((nout,) + self.dims.count)[::-1]).T

        if transpose:
            # swap arrays so that in-memory representation matches that
            # of original input. default is to return array whose shape matches
            # that of the series dims object.
            if size(self.dims.count) == 3:
                out = out.transpose([0, 3, 2, 1])
            if size(self.dims.count) == 2:  # (b, x, y) -> (b, y, x)
                out = out.transpose([0, 2, 1])

        return out.squeeze()

    def subset(self, nsamples=100, thresh=None, stat='std'):
        """Extract random subset of records from a Series,
        filtering on the standard deviation

        Parameters
        ----------
        nsamples : int, optional, default = 100
            The number of data points to sample

        thresh : float, optional, default = None
            A threshold on standard deviation to use when picking points

        stat : str, optional, default = 'std'
            Statistic to use for thresholding

        Returns
        -------
        result : array
            A local numpy array with the subset of points
        """
        from numpy.linalg import norm
        from numpy.random import randint

        statDict = {'std': std, 'norm': norm}
        seed = randint(0, 2 ** 32 - 1)

        if thresh is not None:
            func = statDict[stat]
            result = array(self.rdd.values().filter(lambda x: func(x) > thresh).takeSample(False, nsamples, seed=seed))
        else:
            result = array(self.rdd.values().takeSample(False, nsamples, seed=seed))

        if size(result) == 0:
            raise Exception('No records found, maybe threshold of %g is too high, try changing it?' % thresh)

        return result

    def query(self, inds, var='inds', order='F', isOneBased=True):
        """
        Extract records with indices matching those provided

        Keys will be automatically linearized before matching to provided indices.
        This will not affect

        Parameters
        ----------
        inds : str, or array-like (2D)
            Array of indices, each an array-like of integer indices, or
            filename of a MAT file containing a set of indices as a cell array

        var : str, optional, default = 'inds'
            Variable name if loading from a MAT file

        order : str, optional, default = 'F'
            Specify ordering for linearizing indices (see subtoind)

        onebased : boolean, optional, default = True
            Specify zero or one based indexing for linearizing (see subtoind)

        Returns
        -------
        keys : array, shape (n, k) where k is the length of each value
            Averaged values

        values : array, shape (n, d) where d is the number of keys
            Averaged keys
        """

        if isinstance(inds, str):
            inds = loadMatVar(inds, var)[0]
        else:
            inds = asarray(inds)

        n = len(inds)

        from thunder.rdds.keys import _indToSubConverter
        converter = _indToSubConverter(dims=self.dims.max, order=order, isOneBased=isOneBased)

        keys = zeros((n, len(self.dims.count)))
        values = zeros((n, len(self.first()[1])))

        data = self.subToInd(order=order, isOneBased=isOneBased)

        for idx, indList in enumerate(inds):
            if len(indList) > 0:
                indsSet = set(asarray(indList).flat)
                bcInds = self.rdd.context.broadcast(indsSet)
                values[idx, :] = data.filterOnKeys(lambda k: k in bcInds.value).values().mean()
                keys[idx, :] = mean(map(lambda k: converter(k), indList), axis=0)

        return keys, values

    def __maskToKeys(self, mask, returnNested=False):
        """Helper method to validate and convert a binary mask to a set of keys for use in
        mean of/by region(s).

        If returnNested is true, will return a sequence of sequences of keys, suitable for use
        in meanByRegions. If returnNested is true and an integer or uint mask is passed, a separate set
        of keys will be returned for each unique nonzero value in the mask, sorted in numeric order of the
        mask values.

        If returnNested is false, a single sequence of keys will be returned. The keys will be the indices
        of all nonzero values of the passed mask.

        Parameters
        ----------
        mask: ndarray

        returnNested: boolean, optional, default False

        Returns
        -------
        sequence of subscripted indices if returnNested is false,
        sequence of sequence of subscripted indices if returnNested is true
        """
        # argument type checking
        if not isinstance(mask, ndarray):
            raise ValueError("Mask should be numpy ndarray, got: '%s'" % str(type(mask)))
        # check for matching shapes only if we already know our own shape; don't trigger action otherwise
        # a shape mismatch should be caught downstream, when expected and actual record counts fail to line up
        if self._dims:
            if mask.shape != self._dims.count:
                raise ValueError("Shape mismatch between mask '%s' and series '%s'; shapes must be equal" %
                                 (str(mask.shape), str(self._dims.count)))
        from numpy import nonzero, transpose, unique

        def maskToIndices(bmask):
            return [tuple(idxs) for idxs in transpose(nonzero(bmask))]

        if mask.dtype.kind in ('i', 'u') and returnNested:
            nestedKeys = []
            # integer or unsigned int mask
            for group in unique(mask):
                if group != 0:
                    keys = maskToIndices(mask == group)
                    nestedKeys.append(keys)
            return nestedKeys
        else:
            keys = maskToIndices(mask)
            if returnNested:
                return [keys]
            else:
                return keys

    def meanOfRegion(self, selection, validate=False):
        """Takes the mean of Series values within a single region specified by the passed mask or keys.

        The region for which to take the mean may be specified either by a mask array, or directly by
        Series keys. If an ndarray is passed as `selection`, then the mean will be taken across all series
        records corresponding to nonzero elements of the passed mask. (The passed ndarray must have the
        same shape as series.dims.count, otherwise a ValueError will be thrown.)

        If a sequence of series record keys is passed, the the mean will be taken across all records
        with keys matching one of those in the passed selection sequence.

        `validate` controls checking whether all requested records were included in the calculated mean. If True,
        ValueError will be thrown if the number of records included in the region mean is not equal
        to the number of records specified for that region by the selection. If False, no such checking is performed.

        Parameters
        ----------
        selection: sequence of Series record keys, or ndarray mask

        checkCountMismatch: string "none"|"warn"|"error", or unambiguous prefix ("n","w","e")

        Returns
        -------
        tuple of (tuple(mean of keys), array(mean value)), or (None, None) if no matching records are found
        """

        if isinstance(selection, ndarray):
            selection = self.__maskToKeys(selection, returnNested=False)

        bcRegionKeys = self.rdd.context.broadcast(frozenset(selection))
        n, keyMean, valMean = self.rdd.filter(lambda (k, v): k in bcRegionKeys.value) \
            .map(lambda (k, v):  (array(k, dtype=v.dtype), v)) \
            .aggregate(_MeanCombiner.createZeroTuple(),
                       _MeanCombiner.mergeIntoMeanTuple,
                       _MeanCombiner.combineMeanTuples)
        if isinstance(keyMean, ndarray):
            keyMean = tuple(keyMean.astype('int32'))

        if validate and n != len(selection):
            raise ValueError("%d records were expected in region, but only %d were found" % (len(selection), n))

        return (keyMean, valMean) if n > 0 else (None, None)

    def meanByRegion(self, nestedKeys, validate=False):
        """Takes the mean of Series values within groupings specified by the passed keys.

        Each sequence of keys passed specifies a "region" within which to calculate the mean. For instance,
        series.meanByRegion([[(1,0), (2,0)]) would return the mean of the records in series with keys (1,0) and (2,0).
        If multiple regions are passed in, then multiple aggregates will be returned. For instance,
        series.meanByRegion([[(1,0), (2,0)], [(1,0), (3,0)]]) would return two means, one for the region composed
        of records (1,0) and (2,0), the other for records (1,0) and (3,0).

        Alternatively, an ndarray mask may be passed instead of a sequence of sequences of keys. The array mask
        must be the same shape as the underlying series data (that is, nestedKeys.shape == series.dims.count must
        be True). If an integer or unsigned integer mask is passed, then each unique nonzero element in the passed
        mask will be interpreted as a separate region (that is, all '1's will be a single region, as will all '2's,
        and so on). If another type of ndarray is passed, then all nonzero mask elements will be interpreted
        as a single region.

        This method returns a new Series object, with one record per defined region. Record keys will be the mean of
        keys within the region, while record values will be the mean of values in the region. The `dims` attribute on
        the new Series will not be set; all other attributes will be as in the source Series object.

        `validate` controls checking whether all requested records were included in the calculated mean. If True,
        exceptions will be thrown on the workers if the number of records included in the region mean is not equal
        to the number of records specified for that region by the selection. If False, no such checking is performed.

        Parameters
        ----------
        nestedKeys: sequence of sequences of Series record keys, or ndarray mask.

        validate: boolean, default False

        Returns
        -------
        new Series object
        """
        if isinstance(nestedKeys, ndarray):
            nestedKeys = self.__maskToKeys(nestedKeys, returnNested=True)

        # transform keys into map from keys to sequence of region indices
        regionLookup = {}
        nRecsInRegion = []
        for regionIdx, region in enumerate(nestedKeys):
            nRecsInRegion.append(len(region))
            for key in region:
                regionLookup.setdefault(tuple(key), []).append(regionIdx)

        bcRegionLookup = self.rdd.context.broadcast(regionLookup)

        def toRegionIdx(kvIter):
            regionLookup_ = bcRegionLookup.value
            for k, val in kvIter:
                for regionIdx_ in regionLookup_.get(k, []):
                    yield regionIdx_, (k, val)

        def validateCounts(region_, n_, keyMean, valMean):
            # nRecsInRegion pulled in via closure
            if nRecsInRegion[region_] != n_:
                raise ValueError("%d records were expected in region %d, but only %d were found" %
                                 (nRecsInRegion[region_], region_, n_))
            else:
                return keyMean.astype('int16'), valMean

        combinedData = self.rdd.mapPartitions(toRegionIdx) \
            .combineByKey(_MeanCombiner.createMeanTuple,
                          _MeanCombiner.mergeIntoMeanTuple,
                          _MeanCombiner.combineMeanTuples, numPartitions=len(nestedKeys))

        if validate:
            data = combinedData.map(lambda (region_, (n, keyMean, valMean)):
                                    validateCounts(region_, n, keyMean, valMean))
        else:
            data = combinedData.map(lambda (region_, (_, keyMean, valMean)):
                                    (tuple(keyMean.astype('int16')), valMean))
        return self._constructor(data).__finalize__(self, noPropagate=('_dims',))

    def toBlocks(self, blockSizeSpec="150M"):
        """
        Parameters
        ----------
        blockSizeSpec: string memory size, tuple of integer splits per dimension, or instance of BlockingStrategy
            A string spec will be interpreted as a memory size string (e.g. "64M"). The resulting blocks will be
            generated by a SeriesBlockingStrategy to be close to the requested size.
            A tuple of positive ints will be interpreted as "splits per dimension". Only certain patterns of splits
            are valid to convert Series back to Blocks; see docstring above. These splits will be passed into a
            SeriesBlockingStrategy that will be used to generate the returned blocks.
            If an instance of SeriesBlockingStrategy is passed, it will be used to generate the returned Blocks.

        Returns
        -------
        Blocks instance
        """
        from thunder.rdds.imgblocks.strategy import BlockingStrategy, SeriesBlockingStrategy
        if isinstance(blockSizeSpec, SeriesBlockingStrategy):
            blockingStrategy = blockSizeSpec
        elif isinstance(blockSizeSpec, basestring) or isinstance(blockSizeSpec, int):
            blockingStrategy = SeriesBlockingStrategy.generateFromBlockSize(self, blockSizeSpec)
        else:
            # assume it is a tuple of positive int specifying splits
            blockingStrategy = SeriesBlockingStrategy(blockSizeSpec)

        blockingStrategy.setSource(self)
        avgSize = blockingStrategy.calcAverageBlockSize()
        if avgSize >= BlockingStrategy.DEFAULT_MAX_BLOCK_SIZE:
            # TODO: use logging module here rather than print
            print "Thunder WARNING: average block size of %g bytes exceeds suggested max size of %g bytes" % \
                  (avgSize, BlockingStrategy.DEFAULT_MAX_BLOCK_SIZE)

        returnType = blockingStrategy.getBlocksClass()
        blockedRdd = self.rdd.map(blockingStrategy.blockingFunction)
        # since our blocks are likely pretty big, try setting 1 partition per block
        groupedRdd = blockedRdd.groupByKey(numPartitions=blockingStrategy.nblocks)
        # <key>, <val> at this point is:
        # <block number>, <[(series key, series val), (series key, series val), ...]>
        simpleBlocksRdd = groupedRdd.map(blockingStrategy.combiningFunction)
        return returnType(simpleBlocksRdd, dims=self.dims, nimages=len(self.index), dtype=self.dtype,
                          awsCredentials=self._awsCredentials)

    def saveAsBinarySeries(self, outputdirname, overwrite=False):
        """Writes out Series-formatted data.

        This method (Series.saveAsBinarySeries) writes out binary series files using the current partitioning
        of this Series object. (That is, if mySeries.rdd.getNumPartitions() == 5, then 5 files will be written
        out, one per partition.) The records will not be resorted; the file names for each partition will be
        taken from the key of the first Series record in that partition. If the Series object is already
        sorted and no records have been removed by filtering, then the resulting output should be equivalent
        to what one would get from calling myImages.saveAsBinarySeries().

        If all one wishes to do is to save out Images data in a binary series format, then
        tsc.convertImagesToSeries() will likely be more efficient than
        tsc.loadImages().toSeries().saveAsBinarySeries().

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
        import cStringIO as StringIO
        import struct
        from thunder.rdds.imgblocks.blocks import SimpleBlocks
        from thunder.rdds.fileio.writers import getParallelWriterForPath
        from thunder.rdds.fileio.seriesloader import writeSeriesConfig

        if not overwrite:
            from thunder.utils.common import raiseErrorIfPathExists
            raiseErrorIfPathExists(outputdirname, awsCredentialsOverride=self._awsCredentials)
            overwrite = True  # prevent additional downstream checks for this path

        def partitionToBinarySeries(kvIter):
            """Collects all Series records in a partition into a single binary series record.
            """
            keypacker = None
            firstKey = None
            buf = StringIO.StringIO()
            for seriesKey, series in kvIter:
                if keypacker is None:
                    keypacker = struct.Struct('h'*len(seriesKey))
                    firstKey = seriesKey
                # print >> sys.stderr, seriesKey, series, series.tostring().encode('hex')
                buf.write(keypacker.pack(*seriesKey))
                buf.write(series.tostring())
            val = buf.getvalue()
            buf.close()
            # we might have an empty partition, in which case firstKey will still be None
            if firstKey is None:
                return iter([])
            else:
                label = SimpleBlocks.getBinarySeriesNameForKey(firstKey) + ".bin"
                return iter([(label, val)])

        writer = getParallelWriterForPath(outputdirname)(outputdirname, overwrite=overwrite,
                                                         awsCredentialsOverride=self._awsCredentials)

        binseriesrdd = self.rdd.mapPartitions(partitionToBinarySeries)

        binseriesrdd.foreach(writer.writerFcn)

        # TODO: all we really need here are the number of keys and number of values, which could in principle
        # be cached in _nkeys and _nvals attributes, removing the need for this .first() call in most cases.
        firstKey, firstVal = self.first()
        writeSeriesConfig(outputdirname, len(firstKey), len(firstVal), keyType='int16', valueType=self.dtype,
                          overwrite=overwrite, awsCredentialsOverride=self._awsCredentials)

    def toRowMatrix(self):
        """
        Convert Series to RowMatrix
        """
        from thunder.rdds.matrices import RowMatrix
        return RowMatrix(self.rdd).__finalize__(self)

    def toTimeSeries(self):
        """
        Convert Series to TimeSeries
        """
        from thunder.rdds.timeseries import TimeSeries
        return TimeSeries(self.rdd).__finalize__(self)

    def toSpatialSeries(self):
        """
        Convert Series to SpatialSeries
        """
        from thunder.rdds.spatialseries import SpatialSeries
        return SpatialSeries(self.rdd).__finalize__(self)


class _MeanCombiner(object):
    @staticmethod
    def createZeroTuple():
        return 0, array((0.0,)), array((0.0,))

    @staticmethod
    def createMeanTuple(kv):
        key, val = kv
        return 1, array(key, dtype=val.dtype), val

    @staticmethod
    def mergeIntoMeanTuple(meanTuple, kv):
        n, kmu, vmu = meanTuple
        newn = n+1
        return newn, kmu + (kv[0] - kmu) / newn, vmu + (kv[1] - vmu) / newn

    @staticmethod
    def combineMeanTuples(meanTup1, meanTup2):
        n1, kmu1, vmu1 = meanTup1
        n2, kmu2, vmu2 = meanTup2
        if n1 == 0:
            return n2, kmu2, vmu2
        elif n2 == 0:
            return n1, kmu1, vmu1
        else:
            newn = n1 + n2
            if n2 * 10 < n1:
                kdel = kmu2 - kmu1
                vdel = vmu2 - vmu1
                kmu1 += (kdel * n2) / newn
                vmu1 += (vdel * n2) / newn
            elif n1 * 10 < n2:
                kdel = kmu2 - kmu1
                vdel = vmu2 - vmu1
                kmu1 = kmu2 - (kdel * n1) / newn
                vmu1 = vmu2 - (vdel * n1) / newn
            else:
                kmu1 = (kmu1 * n1 + kmu2 * n2) / newn
                vmu1 = (vmu1 * n1 + vmu2 * n2) / newn
            return newn, kmu1, vmu1