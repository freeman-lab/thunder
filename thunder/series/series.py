from numpy import array, mean, median, std, size, arange, percentile,\
    asarray, zeros, corrcoef, where, unique, array_equal, delete, \
    ravel, logical_not, unravel_index, prod, random, shape, \
    dot, outer, expand_dims, ScalarType, ndarray, sqrt, pi, angle, fft, \
    roll, polyfit, polyval, ceil, float64, fix
import logging
from itertools import product
from bolt.utils import tupleize
from six import string_types
from ..utils import check_options


from ..base import Data


class Series(Data):
    """
    Collection of indexed 1d array data.

    Backed by an array-like object, including a numpy array
    (for local computation) or a bolt array (for spark computation).

    Attributes
    ----------
    values : array-like
        numpy array or bolt array

    index : array-like or one-dimensional list
        Values must be unique, same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    labels : array-like
        A set of labels, one per series record.
    """
    _metadata = Data._metadata
    _attributes = Data._attributes + ['index']

    def __init__(self, values, index=None, labels=None, mode='local'):
        super(Series, self).__init__(values, mode=mode)
        self.labels = labels
        self._index = None
        if index is not None:
            self._index = index

    @property
    def index(self):
        if self._index is None:
            self._index = arange(self.shape[-1])
        return self._index

    @index.setter
    def index(self, value):
        lenself = len(self.index)
        if type(value) is str:
            value = [value]
        try:
            value[0]
        except:
            value = [value]
        try:
            lenvalue = len(value)
        except:
            raise TypeError('Index must be an object with a length')
        if lenvalue != lenself:
            raise ValueError("Length of new index '%g' must match length of original index '%g'"
                             .format(lenvalue, lenself))
        self._index = value

    @property
    def length(self):
        return len(self.index)

    @property
    def baseaxes(self):
        return tuple(range(0, len(self.shape)-1))

    @property
    def _constructor(self):
        return Series

    def flatten(self):
        """
        Reshape all dimensions but the last into a single dimension
        """
        size = prod([s for i, s in enumerate(self.shape) if i in self.baseaxes])
        newvalues = self.values.reshape(size, self.shape[-1])

        if self.labels is not None:
            fullshape = prod(self.labels.shape)
            newlabels = self.labels.reshape(size, fullshape/size).squeeze()
        else:
            newlabels = None

        return self._constructor(newvalues, labels=newlabels).__finalize__(self, noprop=('labels',))

    def count(self):
        """
        Count the number of records.

        For lazy or distributed data, will force a computation.
        """
        if self.mode == 'local':
            return prod(self.shape[:-1])

        if self.mode == 'spark':
            return self.tordd().count()

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'local':
            return self.values[tuple(zeros(len(self.baseaxes))) + (slice(None, None),)]

        if self.mode == 'spark':
            return self.values.tordd().values().first()

    def tolocal(self):
        """
        Convert to local mode.
        """
        from thunder.series.readers import fromarray

        if self.mode == 'local':
            logging.getLogger('thunder').warn('images already in local mode')
            pass

        return fromarray(self.toarray(), index=self.index, labels=self.labels)

    def tospark(self, engine=None):
        """
        Convert to spark mode.
        """
        from thunder.series.readers import fromarray

        if self.mode == 'spark':
            logging.getLogger('thunder').warn('images already in local mode')
            pass

        if engine is None:
            raise ValueError('Must provide SparkContext')

        return fromarray(self.toarray(), index=self.index, labels=self.lables, engine=engine)

    def sample(self, n=100, seed=None):
        """
        Extract random sample of records.

        Parameters
        ----------
        n : int, optional, default = 100
            The number of data points to sample.

        seed : int, optional, default = None
            Random seed.
        """
        if n < 1:
            raise ValueError("Number of samples must be larger than 0, got '%g'" % n)

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if self.mode == 'spark':
            result = asarray(self.values.tordd().values().takeSample(False, n, seed))

        else:
            basedims = [self.shape[d] for d in self.baseaxes]
            inds = [unravel_index(int(k), basedims) for k in random.rand(n) * prod(basedims)]
            result = asarray([self.values[tupleize(i) + (slice(None, None),)] for i in inds])

        return self._constructor(result, index=self.index)

    def map(self, func, index=None, with_keys=False):
        """
        Map an array -> array function over each record.

        Parameters
        ----------
        func : function
            A function of a single record.

        index : array-like, optional, default = None
            If known, the index to be used following function evaluation.

        with_keys : boolean, optional, default = False
            If true, function should be of both tuple indices and series values.
        """
        value_shape = len(index) if index is not None else None
        new = self._map(func, axis=self.baseaxes, value_shape=value_shape, with_keys=with_keys)
        return self._constructor(new.values, index=index, labels=self.labels)

    def reduce(self, func):
        """
        Reduce a function over records.

        Parameters
        ----------
        func : function
            A function of two records.
        """
        return self._reduce(func, axis=self.baseaxes)

    def mean(self):
        """
        Compute the mean across records
        """
        return self._constructor(self.values.mean(axis=self.baseaxes, keepdims=True))

    def var(self):
        """
        Compute the variance across records
        """
        return self._constructor(self.values.var(axis=self.baseaxes, keepdims=True))

    def std(self):
        """
        Compute the standard deviation across records.
        """
        return self._constructor(self.values.std(axis=self.baseaxes, keepdims=True))

    def sum(self):
        """
        Compute the sum across records.
        """
        return self._constructor(self.values.sum(axis=self.baseaxes, keepdims=True))

    def max(self):
        """
        Compute the max across records.
        """
        return self._constructor(self.values.max(axis=self.baseaxes, keepdims=True))

    def min(self):
        """
        Compute the min across records.
        """
        return self._constructor(self.values.min(axis=self.baseaxes, keepdims=True))

    def between(self, left, right):
        """
        Select subset of values within the given index range.

        Inclusive on the left; exclusive on the right.

        Parameters
        ----------
        left : int
            Left-most index in the desired range.

        right: int
            Right-most index in the desired range.
        """
        crit = lambda x: left <= x < right
        return self.select(crit)

    def select(self, crit):
        """
        Select subset of values that match a given index criterion.

        Parameters
        ----------
        crit : function, list, str, int
            Criterion function to map to indices, specific index value,
            or list of indices.
        """
        import types

        # handle lists, strings, and ints
        if not isinstance(crit, types.FunctionType):
            # set("foo") -> {"f", "o"}; wrap in list to prevent:
            if isinstance(crit, string_types):
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
            if crit(index[0]):
                return self
            else:
                raise Exception('No indices found matching criterion')

        # determine new index and check the result
        newindex = [i for i in index if crit(i)]
        if len(newindex) == 0:
            raise Exception('No indices found matching criterion')
        if array(newindex == index).all():
            return self

        # use fast logical indexing to get the new values
        subinds = where([crit(i) for i in index])
        new = self.map(lambda x: x[subinds], index=newindex)

        # if singleton, need to check whether it's an array or a scalar/int
        # if array, recompute a new set of indices
        if len(newindex) == 1:
            new = new.map(lambda x: x[0], index=newindex)
            val = new.first()
            if size(val) == 1:
                newindex = [newindex[0]]
            else:
                newindex = arange(0, size(val))

        new._index = newindex

        return new

    def center(self, axis=1):
        """
        Subtract the mean either within or across records.

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, within (1) or across (0) records.
        """
        if axis == 1:
            return self.map(lambda x: x - mean(x))
        elif axis == 0:
            meanval = self.mean().toarray()
            return self.map(lambda x: x - meanval)
        else:
            raise Exception('Axis must be 0 or 1')

    def standardize(self, axis=1):
        """
        Divide by standard deviation either within or across records.

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to standardize along, within (1) or across (0) records
        """
        if axis == 1:
            return self.map(lambda x: x / std(x))
        elif axis == 0:
            stdval = self.std().toarray()
            return self.map(lambda x: x / stdval)
        else:
            raise Exception('Axis must be 0 or 1')

    def zscore(self, axis=1):
        """
        Subtract the mean and divide by standard deviation within or across records.

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, within (1) or across (0) records
        """
        if axis == 1:
            return self.map(lambda x: (x - mean(x)) / std(x))
        elif axis == 0:
            meanval = self.mean().toarray()
            stdval = self.std().toarray()
            return self.map(lambda x: (x - meanval) / stdval)
        else:
            raise Exception('Axis must be 0 or 1')

    def squelch(self, threshold):
        """
        Set all records that do not exceed the given threhsold to 0.

        Parameters
        ----------
        threshold : scalar
            Level below which to set records to zero
        """
        func = lambda x: zeros(x.shape) if max(x) < threshold else x
        return self.map(func)

    def correlate(self, signal):
        """
        Correlate records against one or many one-dimensional arrays.

        Parameters
        ----------
        signal : array-like
            One or more signals to correlate against.
        """
        s = asarray(signal)

        if s.ndim == 1:
            if size(s) != self.shape[-1]:
                raise ValueError("Length of signal '%g' does not match record length '%g'"
                                 % (size(s), self.shape[-1]))

            return self.map(lambda x: corrcoef(x, s)[0, 1], index=[1])

        elif s.ndim == 2:
            if s.shape[1] != self.shape[-1]:
                raise ValueError("Length of signal '%g' does not match record length '%g'"
                                 % (s.shape[1], self.shape[-1]))
            newindex = arange(0, s.shape[0])
            return self.map(lambda x: array([corrcoef(x, y)[0, 1] for y in s]), index=newindex)

        else:
            raise Exception('Signal to correlate with must have 1 or 2 dimensions')

    def _check_panel(self, length):
        """
        Check that given fixed panel length evenly divides index.

        Parameters
        ----------
        length : int
            Fixed length with which to subdivide index
        """
        n = len(self.index)
        if divmod(n, length)[1] != 0:
            raise ValueError("Panel length '%g' must evenly divide length of series '%g'"
                             % (length, n))
        if n == length:
            raise ValueError("Panel length '%g' cannot be length of series '%g'"
                             % (length, n))

    def mean_by_panel(self, length):
        """
        Compute the mean across fixed sized panels of each record.

        Splits each record into panels of size `length`,
        and then computes the mean across panels.
        Panel length must subdivide record exactly.

        Parameters
        ----------
        length : int
            Fixed length with which to subdivide.
        """
        self._check_panel(length)
        func = lambda v: v.reshape(-1, length).mean(axis=0)
        newindex = arange(length)
        return self.map(func, index=newindex)

    def _makemasks(self, index=None, level=0):
        """
        Internal function for generating masks for selecting values based on multi-index values.

        As all other multi-index functions will call this function, basic type-checking is also
        performed at this stage.
        """
        if index is None:
            index = self.index

        try:
            dims = len(array(index).shape)
            if dims == 1:
                index = array(index, ndmin=2).T
        except:
            raise TypeError('A multi-index must be convertible to a numpy ndarray')

        try:
            index = index[:, level]
        except:
            raise ValueError("Levels must be indices into individual elements of the index")

        lenIdx = index.shape[0]
        nlevels = index.shape[1]

        combs = product(*[unique(index.T[i, :]) for i in range(nlevels)])
        combs = array([l for l in combs])

        masks = array([[array_equal(index[i], c) for i in range(lenIdx)] for c in combs])

        return zip(*[(masks[x], combs[x]) for x in range(len(masks)) if masks[x].any()])

    def _map_by_index(self, function, level=0):
        """
        An internal function for maping a function to groups of values based on a multi-index

        Elements of each record are grouped according to unique value combinations of the multi-
        index across the given levels of the multi-index. Then the given function is applied
        to to each of these groups separately. If this function is many-to-one, the result
        can be recast as a Series indexed by the unique index values used for grouping.
        """

        if type(level) is int:
            level = [level]

        masks, ind = self._makemasks(index=self.index, level=level)
        nMasks = len(masks)
        newindex = array(ind)
        if len(newindex[0]) == 1:
            newindex = ravel(newindex)
        return self.map(lambda v: asarray([array(function(v[masks[x]])) for x in range(nMasks)]),
                        index=newindex)

    def select_by_index(self, val, level=0, squeeze=False, filter=False, return_mask=False):
        """
        Select or filter elements of the Series by index values (across levels, if multi-index).

        The index is a property of a Series object that assigns a value to each position within
        the arrays stored in the records of the Series. This function returns a new Series where,
        within each record, only the elements indexed by a given value(s) are retained. An index
        where each value is a list of a fixed length is referred to as a 'multi-index',
        as it provides multiple labels for each index location. Each of the dimensions in these
        sublists is a 'level' of the multi-index. If the index of the Series is a multi-index, then
        the selection can proceed by first selecting one or more levels, and then selecting one
        or more values at each level.

        Parameters
        ----------
        val : list of lists
            Specifies the selected index values. List must contain one list for each level of the
            multi-index used in the selection. For any singleton lists, the list may be replaced
            with just the integer.

        level : list of ints, optional, default=0
            Specifies which levels in the multi-index to use when performing selection. If a single
            level is selected, the list can be replaced with an integer. Must be the same length
            as val.

        squeeze : bool, optional, default=False
            If True, the multi-index of the resulting Series will drop any levels that contain
            only a single value because of the selection. Useful if indices are used as unique
            identifiers.

        filter : bool, optional, default=False
            If True, selection process is reversed and all index values EXCEPT those specified
            are selected.

        return_mask : bool, optional, default=False
            If True, return the mask used to implement the selection.
        """
        try:
            level[0]
        except:
            level = [level]
        try:
            val[0]
        except:
            val = [val]

        remove = []
        if len(level) == 1:
            try:
                val[0][0]
            except:
                val = [val]
            if squeeze and not filter and len(val) == 1:
                remove.append(level[0])
        else:
            for i in range(len(val)):
                try:
                    val[i][0]
                except:
                    val[i] = [val[i]]
                if squeeze and not filter and len(val[i]) == 1:
                    remove.append(level[i])

        if len(level) != len(val):
            raise ValueError("List of levels must be same length as list of corresponding values")

        p = product(*val)
        selected = set([x for x in p])

        masks, ind = self._makemasks(index=self.index, level=level)
        nmasks = len(masks)
        masks = array([masks[x] for x in range(nmasks) if tuple(ind[x]) in selected])

        final_mask = masks.any(axis=0)
        if filter:
            final_mask = logical_not(final_mask)

        indFinal = array(self.index)
        if len(indFinal.shape) == 1:
            indFinal = array(indFinal, ndmin=2).T
        indFinal = indFinal[final_mask]

        if squeeze:
            indFinal = delete(indFinal, remove, axis=1)

        if len(indFinal[0]) == 1:
            indFinal = ravel(indFinal)

        elif len(indFinal[1]) == 0:
            indFinal = arange(sum(final_mask))

        result = self.map(lambda v: v[final_mask], index=indFinal)

        if return_mask:
            return result, final_mask
        else:
            return result

    def aggregate_by_index(self, function, level=0):
        """
        Aggregrate data in each record, grouping by index values.

        For each unique value of the index, applies a function to the group
        indexed by that value. Returns a Series indexed by those unique values.
        For the result to be a valid Series object, the aggregating function should
        return a simple numeric type. Also allows selection of levels within a
        multi-index. See select_by_index for more info on indices and multi-indices.

        Parameters
        ----------
        function : function
            Aggregating function to map to Series values. Should take a list or ndarray
            as input and return a simple numeric value.

        level : list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values.
            If only a single level is desired, can be an int.
        """
        result = self._map_by_index(function, level=level)
        return result.map(lambda v: array(v), index=result.index)

    def stat_by_index(self, stat, level=0):
        """
        Compute the desired statistic for each uniue index values (across levels, if multi-index)

        Parameters
        ----------
        stat : string
            Statistic to be computed: sum, mean, median, stdev, max, min, count

        level : list of ints, optional, default=0
            Specifies the levels of the multi-index to use when determining unique index values.
            If only a single level is desired, can be an int.
        """
        from numpy import sum, min, max

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
        return self.aggregate_by_index(level=level, function=func)

    def sum_by_index(self, level=0):
        """
        Compute sums for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='sum')

    def mean_by_index(self, level=0):
        """
        Compute means for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='mean')

    def median_by_index(self, level=0):
        """
        Compute medians for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='median')

    def std_by_index(self, level=0):
        """
        Compute means for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='stdev')

    def max_by_index(self, level=0):
        """
        Compute maximum values for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='max')

    def min_by_index(self, level=0):
        """
        Compute minimum values for each unique index value (across level, if multi-index)
        """
        return self.stat_by_index(level=level, stat='min')

    def count_by_index(self, level=0):
        """
        Count the number for each unique index value (across levels, if multi-index)
        """
        return self.stat_by_index(level=level, stat='count')

    def cov(self):
        """
        Compute covariance of a distributed matrix.

        Parameters
        ----------
        axis : int, optional, default = None
            Axis for performing mean subtraction, None (no subtraction), 0 (rows) or 1 (columns)
        """
        return self.center(axis=0).gramian().times(1.0 / (self.shape[0] - 1))

    def gramian(self):
        """
        Compute gramian of a distributed matrix.

        The gramian is defined as the product of the matrix
        with its transpose, i.e. A^T * A.
        """
        if self.mode == 'spark':
            rdd = self.values.tordd()

            from pyspark.accumulators import AccumulatorParam

            class MatrixAccumulator(AccumulatorParam):
                def zero(self, value):
                    return zeros(shape(value))

                def addInPlace(self, val1, val2):
                    val1 += val2
                    return val1

            global mat
            init = zeros((self.shape[1], self.shape[1]))
            mat = rdd.context.accumulator(init, MatrixAccumulator())

            def outer_sum(x):
                global mat
                mat += outer(x, x)

            rdd.values().foreach(outer_sum)
            return self._constructor(mat.value, index=self.index)

        if self.mode == 'local':
            return self._constructor(dot(self.values.T, self.values), index=self.index)

    def times(self, other):
        """
        Multiply a matrix by another one.

        Other matrix must be a numpy array, a scalar,
        or another matrix in local mode.

        Parameters
        ----------
        other : Matrix, scalar, or numpy array
            A matrix to multiply with
        """
        if isinstance(other, ScalarType):
            other = asarray(other)
            index = self.index
        else:
            if isinstance(other, list):
                other = asarray(other)
            if isinstance(other, ndarray) and other.ndim < 2:
                other = expand_dims(other, 1)
            if not self.shape[1] == other.shape[0]:
                raise ValueError('shapes %s and %s are not aligned' % (self.shape, other.shape))
            index = arange(other.shape[1])

        if self.mode == 'local' and isinstance(other, Series) and other.mode == 'spark':
            raise NotImplementedError

        if self.mode == 'spark' and isinstance(other, Series) and other.mode == 'spark':
            raise NotImplementedError

        if self.mode == 'local' and isinstance(other, (ndarray, ScalarType)):
            return self._constructor(dot(self.values, other), index=index)

        if self.mode == 'local' and isinstance(other, Series):
            return self._constructor(dot(self.values, other.values), index=index)

        if self.mode == 'spark' and isinstance(other, (ndarray, ScalarType)):
            return self.map(lambda x: dot(x, other), index=index)

        if self.mode == 'spark' and isinstance(other, Series):
            return self.map(lambda x: dot(x, other.values), index=index)

    def _makewindows(self, indices, window):
        """
        Make masks used by windowing functions

        Given a list of indices specifying window centers,
        and a window size, construct a list of index arrays,
        one per window, that index into the target array

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        div = divmod(window, 2)
        before = div[0]
        after = div[0] + div[1]
        index = asarray(self.index)
        indices = asarray(indices)
        if where(index == max(indices))[0][0] + after > len(index):
            raise ValueError("Maximum requested index %g, with window %g, exceeds length %g"
                             % (max(indices), window, len(index)))
        if where(index == min(indices))[0][0] - before < 0:
            raise ValueError("Minimum requested index %g, with window %g, is less than 0"
                             % (min(indices), window))
        masks = [arange(where(index == i)[0][0]-before, where(index == i)[0][0]+after, dtype='int') for i in indices]
        return masks

    def mean_by_window(self, indices, window):
        """
        Average series across multiple windows specified by their centers.

        Parameters
        ----------
        indices : array-like
            List of times specifying window centers

        window : int
            Window size
        """
        masks = self._makewindows(indices, window)
        newindex = arange(0, len(masks[0]))
        return self.map(lambda x: mean([x[m] for m in masks], axis=0), index=newindex)

    def subsample(self, sampleFactor=2):
        """
        Subsample series by an integer factor.

        Parameters
        ----------
        sampleFactor : positive integer, optional, default=2

        """
        if sampleFactor < 0:
            raise Exception('Factor for subsampling must be postive, got %g' % sampleFactor)
        s = slice(0, len(self.index), sampleFactor)
        newindex = self.index[s]
        return self.map(lambda v: v[s], index=newindex)

    def fourier(self, freq=None):
        """
        Compute statistics of a Fourier decomposition on series data.

        Parameters
        ----------
        freq : int
            Digital frequency at which to compute coherence and phase
        """
        def get(y, freq):
            y = y - mean(y)
            nframes = len(y)
            ft = fft.fft(y)
            ft = ft[0:int(fix(nframes/2))]
            ampFt = 2*abs(ft)/nframes
            amp = ampFt[freq]
            ampSum = sqrt(sum(ampFt**2))
            co = amp / ampSum
            ph = -(pi/2) - angle(ft[freq])
            if ph < 0:
                ph += pi * 2
            return array([co, ph])

        if freq >= int(fix(size(self.index)/2)):
            raise Exception('Requested frequency, %g, is too high, '
                            'must be less than half the series duration' % freq)

        index = ['coherence', 'phase']
        return self.map(lambda x: get(x, freq), index=index)

    def convolve(self, signal, mode='full'):
        """
        Conolve series data against another signal.

        Parameters
        ----------
        signal : array
            Signal to convolve with (must be 1D)

        var : str
            Variable name if loading from a MAT file

        mode : str, optional, default='full'
            Mode of convolution, options are 'full', 'same', and 'same'
        """

        from numpy import convolve

        s = asarray(signal)

        n = size(self.index)
        m = size(s)

        # use expected lengths to make a new index
        if mode == 'same':
            newmax = max(n, m)
        elif mode == 'valid':
            newmax = max(m, n) - min(m, n) + 1
        else:
            newmax = n+m-1
        newindex = arange(0, newmax)

        return self.map(lambda x: convolve(x, signal, mode), index=newindex)

    def crosscorr(self, signal, lag=0):
        """
        Cross correlate series data against another signal.

        Parameters
        ----------
        signal : array
            Signal to correlate against (must be 1D).

        lag : int
            Range of lags to consider, will cover (-lag, +lag).
        """
        from scipy.linalg import norm

        s = asarray(signal)
        s = s - mean(s)
        s = s / norm(s)

        if size(s) != size(self.index):
            raise Exception('Size of signal to cross correlate with, %g, '
                            'does not match size of series' % size(s))

        # created a matrix with lagged signals
        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(s)
            m = len(shifts)
            sshifted = zeros((m, d))
            for i in range(0, len(shifts)):
                tmp = roll(s, shifts[i])
                if shifts[i] < 0:
                    tmp[(d+shifts[i]):] = 0
                if shifts[i] > 0:
                    tmp[:shifts[i]] = 0
                sshifted[i, :] = tmp
            s = sshifted
        else:
            shifts = [0]

        def get(y, s):
            y = y - mean(y)
            n = norm(y)
            if n == 0:
                b = zeros((s.shape[0],))
            else:
                y /= norm(y)
                b = dot(s, y)
            return b

        return self.map(lambda x: get(x, s), index=shifts)

    def detrend(self, method='linear', order=5):
        """
        Detrend series data with linear or nonlinear detrending.

        Preserve intercept so that subsequent operations can adjust the baseline.

        Parameters
        ----------
        method : str, optional, default = 'linear'
            Detrending method

        order : int, optional, default = 5
            Order of polynomial, for non-linear detrending only
        """
        check_options(method, ['linear', 'nonlinear'])

        if method == 'linear':
            order = 1

        def func(y):
            x = arange(len(y))
            p = polyfit(x, y, order)
            p[-1] = 0
            yy = polyval(p, x)
            return y - yy

        return self.map(func)

    def normalize(self, method='percentile', window=None, perc=20, offset=0.1):
        """
        Normalize by subtracting and dividing by a baseline.

        Baseline can be derived from a global mean or percentile,
        or a smoothed percentile estimated within a rolling window.
        Windowed baselines may only be well-defined for
        temporal series data.

        Parameters
        ----------
        baseline : str, optional, default = 'percentile'
            Quantity to use as the baseline, options are 'mean', 'percentile',
            'window', or 'window-exact'.

        window : int, optional, default = 6
            Size of window for baseline estimation,
            for 'window' and 'window-exact' baseline only.

        perc : int, optional, default = 20
            Percentile value to use, for 'percentile',
            'window', or 'window-exact' baseline only.

        offset : float, optional, default = 0.1
             Scalar added to baseline during division to avoid division by 0.
        """

        check_options(method, ['mean', 'percentile', 'window', 'window-exact'])

        from warnings import warn
        if not (method == 'window' or method == 'window-exact') and window is not None:
            warn('Setting window without using method "window" has no effect')

        if method == 'mean':
            baseFunc = mean

        if method == 'percentile':
            baseFunc = lambda x: percentile(x, perc)

        if method == 'window':
            from scipy.ndimage.filters import percentile_filter
            baseFunc = lambda x: percentile_filter(x.astype(float64), perc, window, mode='nearest')

        if method == 'window-exact':
            if window & 0x1:
                left, right = (ceil(window/2), ceil(window/2) + 1)
            else:
                left, right = (window/2, window/2)

            n = len(self.index)
            baseFunc = lambda x: asarray([percentile(x[max(ix-left, 0):min(ix+right+1, n)], perc)
                                          for ix in arange(0, n)])

        def get(y):
            b = baseFunc(y)
            return (y - b) / (b + offset)

        return self.map(get)

    def toimages(self, size='150'):
        """
        Converts Series to Images.

        Equivalent to calling series.toBlocks(size).toImages()

        Parameters
        ----------
        size : str, optional, default = "150M"
            String interpreted as memory size.
        """
        from thunder.images.images import Images

        n = len(self.shape) - 1

        if self.mode == 'spark':
            return Images(self.values.swap(tuple(range(n)), (0,), size=size))

        if self.mode == 'local':
            return Images(self.values.transpose((n,) + tuple(range(0, n))))

    def tobinary(self, path, prefix='series', overwrite=False, credentials=None):
        """
        Write data to binary files.

        Parameters
        ----------
        path : string path or URI to directory to be created
            Output files will be written underneath path.
            Directory will be created as a result of this call.

        prefix : str, optional, default = 'series'
            String prefix for files.

        overwrite : bool
            If true, path and all its contents will be deleted and
            recreated as partof this call.
        """
        from thunder.series.writers import tobinary
        tobinary(self, path, prefix=prefix, overwrite=overwrite, credentials=credentials)
