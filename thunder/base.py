from numpy import array, asarray, ndarray, prod, ufunc, add, subtract, \
    multiply, divide, isscalar, newaxis, unravel_index, dtype
from bolt.utils import inshape, tupleize, slicify
from bolt.base import BoltArray
from bolt.spark.array import BoltArraySpark
from bolt.spark.chunk import ChunkedArray
from functools import reduce


from .utils import notsupported

class Base(object):
    """
    Base methods for data objects in thunder.

    Data objects are backed by array-like objects,
    including numpy arrays (for local computation),
    and bolt arrays (for spark computation).

    Handles construction, metadata, and backend specific methods.
    """
    _metadata = ['dtype', 'shape', 'mode']
    _attributes = []

    def __init__(self, values, mode='local'):
        self._values = values
        if isinstance(values, BoltArray) or isinstance(values, ChunkedArray):
            mode = 'spark'
        if isinstance(values, ndarray):
            mode = 'local'
        self._mode = mode

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%s: %s' % ('mode', getattr(self, 'mode'))
        for k in self._metadata:
            v = getattr(self, k)
            output = str(v)
            if len(output) > 50:
                output = output[0:50].strip() + ' ... '
                if output.lstrip().startswith('['):
                    output += '] '
                if hasattr(v, '__len__'):
                    output += '(length: %d)' % len(v)
            if not k == 'mode':
                s += '\n%s: %s' % (k, output)
        return s

    def __finalize__(self, other, noprop=()):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        noprop : iterable of string attribute names, default empty tuple
            Attributes found in noprop will *not* have their values propagated forward.
        """
        if isinstance(other, Data):
            for name in self._attributes:
                if name not in noprop:
                    attr = getattr(other, name, None)
                    if attr is not None:
                        object.__setattr__(self, name, attr)
        return self

    def __array__(self):
        return self.toarray()

    @property
    def _constructor(self):
        return Data

    @property
    def dtype(self):
        return dtype(self._values.dtype)

    @property
    def shape(self):
        return self._values.shape

    @property
    def mode(self):
        return self._mode

    @property
    def values(self):
        return self._values

    def tordd(self):
        """
        Return an RDD for datasets backed by Spark (Spark only).
        """
        if self.mode == 'spark':
            return self.values.tordd()
        else:
            raise NotImplementedError('Cannot return RDD for local data')

    def compute(self):
        """
        Force lazy computations to execute for datasets backed by Spark (Spark only).
        """
        if self.mode == 'spark':
            self.values.tordd().count()
        else:
            notsupported(self.mode)

    def coalesce(self, npartitions):
        """
        Coalesce data (Spark only).

        Parameters
        ----------
        npartitions : int
            Number of partitions after coalescing.
        """
        if self.mode == 'spark':
            current = self.values.tordd().getNumPartitions()
            if npartitions > current:
                raise Exception('Trying to increase number of partitions (from %g to %g), '
                                'cannot use coalesce, try repartition' % (current, npartitions))
            self.values._rdd = self.values._rdd.coalesce(npartitions)
            return self
        else:
            notsupported(self.mode)

    def cache(self):
        """
        Enable in-memory caching (Spark only).
        """
        if self.mode == 'spark':
            self.values.cache()
            return self
        else:
            notsupported(self.mode)

    def uncache(self):
        """
        Enable in-memory caching.
        """
        if self.mode == 'spark':
            self.values.unpersist()
            return self
        else:
            notsupported(self.mode)

    def iscached(self):
        """
        Get whether object is cached.
        """
        if self.mode == 'spark':
            return self.tordd().iscached
        else:
            notsupported(self.mode)

    def npartitions(self):
        """
        Get number of partitions.
        """
        if self.mode == 'spark':
            return self.tordd().getNumPartitions()
        else:
            notsupported(self.mode)

    def repartition(self, npartitions):
        """
        Repartition data (Spark only).

        Parameters
        ----------
        npartitions : int
            Number of partitions after repartitions.
        """
        if self.mode == 'spark':
            return self._constructor(self.values.repartition(npartitions)).__finalize__(self)
        else:
            notsupported(self.mode)

class Data(Base):
    """
    Generic base class for primary data types in Thunder.

    All data types are backed by an array-like container for many homogenous arrays.
    Data types include Images and Series and their derivatives.

    This class mainly provides convenience functions for accessing
    properties, computing generic summary statistics, and applying
    functions along axes in a backend-specific manner.
    """
    _metadata = Base._metadata
    _attributes = Base._attributes + ['labels']

    def __getitem__(self, item):
        # handle values -- convert ints to slices so no dimensions are dropped
        if isinstance(item, int):
            item = tuple([slicify(item, self.shape[0])])
        if isinstance(item, tuple):
            item = tuple([slicify(i, n) if isinstance(i, int) else i for i, n in zip(item, self.shape[:len(item)])])
        if isinstance(item, (list, ndarray)):
            item = (item,)
        new = self._values.__getitem__(item)
        result = self._constructor(new).__finalize__(self, noprop=('index', 'labels'))

        # handle labels
        if self.labels is not None:
            if isinstance(item, int):
                label_item = ([item],)
            elif isinstance(item, (list, ndarray, slice)):
                label_item = (item, )
            elif isinstance(item, tuple):
                label_item = item[:len(self.baseaxes)]
            newlabels = self.labels
            for (i, s) in enumerate(label_item):
                if isinstance(s, slice):
                    newlabels = newlabels[[s if j==i else slice(None) for j in range(len(label_item))]]
                else:
                    newlabels = newlabels.take(tupleize(s), i)
            result.labels = newlabels

        return result

    @property
    def baseaxes(self):
        raise NotImplementedError

    @property
    def baseshape(self):
        return self.shape[:len(self.baseaxes)]

    @property
    def value_shape(self):
        return self.shape[len(self.baseaxes):]

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is not None:
            try:
                value = asarray(value)
            except:
                raise ValueError("Labels must be convertible to an ndarray")
            if value.shape != self.baseshape:
                raise ValueError("Labels shape {} must be the same as the leading dimensions of the Series {}"\
                                  .format(value.shape, self.baseshape))

        self._labels = value

    def astype(self, dtype, casting='unsafe'):
        """
        Cast values to the specified type.
        """
        return self._constructor(
            self.values.astype(dtype=dtype, casting=casting)).__finalize__(self)

    def toarray(self):
        """
        Return all records to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        return asarray(self.values).squeeze()

    def tospark(self):
        """
        Convert data to Spark.
        """
        raise NotImplementedError

    def tolocal(self):
        """
        Convert data to local mode.
        """
        raise NotImplementedError

    def count(self):
        """
        Explicit count of elements.
        """
        raise NotImplementedError

    def first(self):
        """
        Return first element.
        """
        raise NotImplementedError

    def mean(self):
        """
        Mean of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def sum(self):
        """
        Sum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def var(self):
        """
        Variance of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def std(self):
        """
        Standard deviation computed of values along the appropriate dimension.
        """
        raise NotImplementedError

    def max(self):
        """
        Maximum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def min(self):
        """
        Minimum of values computed along the appropriate dimension.
        """
        raise NotImplementedError

    def map(self, func, **kwargs):
        """
        Map a function over elements.
        """
        raise NotImplementedError

    def _align(self, axes, key_shape=None):
        """
        Align local arrays so that axes for iteration are in the keys.

        This operation is applied before most functional operators.
        It ensures that the specified axes are valid, and might transpose/reshape
        the underlying array so that the functional operators can be applied
        over the correct records.

        Parameters
        ----------
        axes: tuple[int]
            One or more axes that will be iterated over by a functional operator
        """
        # ensure that the key axes are valid for an ndarray of this shape
        inshape(self.shape, axes)

        # compute the set of dimensions/axes that will be used to reshape
        remaining = [dim for dim in range(len(self.shape)) if dim not in axes]
        key_shape = key_shape if key_shape else [self.shape[axis] for axis in axes]
        remaining_shape = [self.shape[axis] for axis in remaining]
        linearized_shape = [prod(key_shape)] + remaining_shape

        # compute the transpose permutation
        transpose_order = list(axes) + remaining

        # transpose the array so that the keys being mapped over come first, then linearize keys
        reshaped = self.values.transpose(*transpose_order).reshape(*linearized_shape)

        return reshaped

    def filter(self, func):
        """
        Filter array along an axis.

        Applies a function which should evaluate to boolean,
        along a single axis or multiple axes. Array will be
        aligned so that the desired set of axes are in the
        keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function to apply, should return boolean

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to filter along.
        """

        if self.mode == 'local':
            reshaped = self._align(self.baseaxes)
            filtered = asarray(list(filter(func, reshaped)))

            if self.labels is not None:
                mask = asarray(list(map(func, reshaped)))

        if self.mode == 'spark':

            sort = False if self.labels is None else True
            filtered = self.values.filter(func, axis=self.baseaxes, sort=sort)

            if self.labels is not None:
                keys, vals = zip(*self.values.map(func, axis=self.baseaxes, value_shape=(1,)).tordd().collect())
                perm = sorted(range(len(keys)), key=keys.__getitem__)
                mask = asarray(vals)[perm]

        if self.labels is not None:
            s1 = prod(self.baseshape)
            newlabels = self.labels.reshape(s1, 1)[mask].squeeze()
        else:
            newlabels = None

        return self._constructor(filtered, labels=newlabels).__finalize__(self, noprop=('labels',))

    def map(self, func, value_shape=None, dtype=None, with_keys=False):
        """
        Apply an array -> array function across an axis.

        Array will be aligned so that the desired set of axes
        are in the keys, which may require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of a single array to apply. If with_keys=True,
            function should be of a (tuple, array) pair.

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to apply function along.

        value_shape : tuple, optional, default=None
            Known shape of values resulting from operation. Only
            valid in spark mode.

        dtype: numpy.dtype, optional, default=None
            Known shape of dtype resulting from operation. Only
            valid in spark mode.

        with_keys : bool, optional, default=False
            Include keys as an argument to the function
        """
        axis = self.baseaxes

        if self.mode == 'local':
            axes = sorted(tupleize(axis))
            key_shape = [self.shape[axis] for axis in axes]
            reshaped = self._align(axes, key_shape=key_shape)

            if with_keys:
                keys = zip(*unravel_index(range(prod(key_shape)), key_shape))
                mapped = asarray(list(map(func, zip(keys, reshaped))))
            else:
                mapped = asarray(list(map(func, reshaped)))

            try:
                elem_shape = mapped[0].shape
            except:
                elem_shape = (1,)

            expand = list(elem_shape)
            expand = [1] if len(expand) == 0 else expand

            # invert the previous reshape operation, using the shape of the map result
            linearized_shape_inv = key_shape + expand
            reordered = mapped.reshape(*linearized_shape_inv)

            return self._constructor(reordered, mode=self.mode).__finalize__(self, noprop=('index'))

        if self.mode == 'spark':
            expand = lambda x: array(func(x), ndmin=1)
            mapped = self.values.map(expand, axis, value_shape, dtype, with_keys)
            return self._constructor(mapped, mode=self.mode).__finalize__(self, noprop=('index',))

    def _reduce(self, func, axis=0):
        """
        Reduce an array along an axis.

        Applies an associative/commutative function of two arguments
        cumulatively to all arrays along an axis. Array will be aligned
        so that the desired set of axes are in the keys, which may
        require a transpose/reshape.

        Parameters
        ----------
        func : function
            Function of two arrays that returns a single array

        axis : tuple or int, optional, default=(0,)
            Axis or multiple axes to reduce along.
        """
        if self.mode == 'local':
            axes = sorted(tupleize(axis))

            # if the function is a ufunc, it can automatically handle reducing over multiple axes
            if isinstance(func, ufunc):
                inshape(self.shape, axes)
                reduced = func.reduce(self, axis=tuple(axes))
            else:
                reshaped = self._align(axes)
                reduced = reduce(func, reshaped)

            # ensure that the shape of the reduced array is valid
            expected_shape = [self.shape[i] for i in range(len(self.shape)) if i not in axes]
            if reduced.shape != tuple(expected_shape):
                raise ValueError("reduce did not yield an array with valid dimensions")

            return self._constructor(reduced[newaxis, :]).__finalize__(self)

        if self.mode == 'spark':
            reduced = self.values.reduce(func, axis, keepdims=True)
            return self._constructor(reduced).__finalize__(self)

    def element_wise(self, other, op):
        """
        Apply an elementwise operation to data.

        Both self and other data must have the same mode.
        If self is in local mode, other can also be a numpy array.
        Self and other must have the same shape, or other must be a scalar.

        Parameters
        ----------
        other : Data or numpy array
           Data to apply elementwise operation to

        op : function
            Binary operator to use for elementwise operations, e.g. add, subtract
        """
        if not isscalar(other) and not self.shape == other.shape:
            raise ValueError("shapes %s and %s must be equal" % (self.shape, other.shape))

        if not isscalar(other) and isinstance(other, Data) and not self.mode == other.mode:
            raise NotImplementedError

        if isscalar(other):
            return self.map(lambda x: op(x, other))

        if self.mode == 'local' and isinstance(other, ndarray):
            return self._constructor(op(self.values, other)).__finalize__(self)

        if self.mode == 'local' and isinstance(other, Data):
            return self._constructor(op(self.values, other.values)).__finalize__(self)

        if self.mode == 'spark' and isinstance(other, Data):

            def func(record):
                (k1, x), (k2, y) = record
                return k1, op(x, y)

            rdd = self.tordd().zip(other.tordd()).map(func)
            barray = BoltArraySpark(rdd, shape=self.shape, dtype=self.dtype, split=self.values.split)
            return self._constructor(barray).__finalize__(self)

    def plus(self, other):
        """
        Elementwise addition.

        See also
        --------
        elementwise
        """
        return self.element_wise(other, add)

    def minus(self, other):
        """
        Elementwise subtraction.

        See also
        --------
        elementwise
        """
        return self.element_wise(other, subtract)

    def dottimes(self, other):
        """
        Elementwise multiplication.

        See also
        --------
        elementwise
        """
        return self.element_wise(other, multiply)

    def dotdivide(self, other):
        """
        Elementwise divison.

        See also
        --------
        elementwise
        """
        return self.element_wise(other, divide)

    def clip(self, min=None, max=None):
        """
        Clip values above and below.

        Parameters
        ----------
        min : scalar or array-like
            Minimum value. If array, will be broadcasted

        max : scalar or array-like
            Maximum value. If array, will be broadcasted.
        """
        return self._constructor(
            self.values.clip(min=min, max=max)).__finalize__(self)
