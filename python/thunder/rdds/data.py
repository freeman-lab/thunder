from numpy import asarray, cumsum, maximum, minimum


class Data(object):
    """
    Generic base class for data types in thunder.

    All data types are backed by an RDD of key-value pairs
    where the key is a tuple identifier and the value is an array

    This base class mainly provides convenience functions for accessing
    properties of the object using the appropriate RDD methods.

    Attributes
    ----------

    `rdd` : Spark RDD
        The Spark Resilient Distributed Dataset wrapped by this Data object.
        Standard pyspark RDD methods on a data instance `obj` that are not already
        directly exposed by the Data object can be accessed via `obj.rdd`.
    """

    _metadata = ['_nrecords', '_dtype']

    def __init__(self, rdd, nrecords=None, dtype=None):
        self.rdd = rdd
        self._nrecords = nrecords
        self._dtype = dtype

    def __repr__(self):
        # start with class name
        s = self.__class__.__name__
        # build a printable string by iterating through _metadata elements
        for k in self._metadata:
            v = getattr(self, k)
            if v is None:
                output = 'None (inspect to compute)'
            else:
                output = str(v)
            # TODO make max line length a configurable property
            if len(output) > 50:
                output = output[0:50].strip() + ' ... '
                if output.lstrip().startswith('['):
                    output += '] '
                if hasattr(v, '__len__'):
                    output += '(length: %d)' % len(v)
            # drop any leading underscores from attribute name:
            if k.startswith('_'):
                k = k.lstrip('_')
            s += '\n%s: %s' % (k, output)
        return s

    @property
    def nrecords(self):
        if self._nrecords is None:
            self._nrecords = self.rdd.count()
        return self._nrecords

    @property
    def dtype(self):
        if not self._dtype:
            self.populateParamsFromFirstRecord()
        return self._dtype

    def populateParamsFromFirstRecord(self):
        """
        Calls first() on the underlying rdd, using the returned record to determine appropriate attribute settings
        for this object (for instance, setting self.dtype to match the dtype of the underlying rdd records).

        This method is expected to be overridden by subclasses. Subclasses should first call
        super(cls, self).populateParamsFromFirstRecord(), then use the returned record to set any additional attributes.

        Returns the result of calling self.rdd.first().
        """

        from numpy import asarray
        
        record = self.rdd.first()
        self._dtype = str(asarray(record[1]).dtype)
        return record

    def __finalize__(self, other, noPropagate=()):
        """
        Lazily propagate attributes from other to self, only if attributes
        are not already defined in self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate

        noPropagate : iterable of string attribute names, default empty tuple
            attributes found in nopropagate will *not* have their values propagated forward from the passed object,
            but will keep their existing values, even if these are None. Attribute names should be specified
            in their "private" versions (with underscores; e.g. "_dtype" and not "dtype") where applicable.
        """
        if isinstance(other, Data):
            for name in self._metadata:
                if name not in noPropagate:
                    otherAttr = getattr(other, name, None)
                    if (otherAttr is not None) and (getattr(self, name, None) is None):
                        object.__setattr__(self, name, otherAttr)
        return self

    @property
    def _constructor(self):
        return Data

    def _resetCounts(self):
        self._nrecords = None
        return self

    def _checkOverwrite(self, outputDirPath):
        """ Checks for existence of outputDirPath, raising ValueError if it already exists """
        from thunder.utils.aws import AWSCredentials
        from thunder.utils.common import raiseErrorIfPathExists
        awsCredentialOverride = AWSCredentials.fromContext(self.rdd.ctx)
        raiseErrorIfPathExists(outputDirPath, awsCredentialsOverride=awsCredentialOverride)

    def first(self):
        """
        Return first record.

        This calls the Spark first() method on the underlying RDD. As a side effect, any attributes on this object that
        can be set based on the values of the first record will be set (see populateParamsFromFirstRecord).
        """
        return self.populateParamsFromFirstRecord()

    def take(self, *args, **kwargs):
        """
        Take samples.

        This calls the Spark take() method on the underlying RDD.
        """
        return self.rdd.take(*args, **kwargs)

    @staticmethod
    def __getKeyTypeCheck(actualKey, keySpec):
        if hasattr(actualKey, "__iter__"):
            try:
                specLen = len(keySpec) if hasattr(keySpec, "__len__") else \
                    reduce(lambda x, y: x + y, [1 for item in keySpec], initial=0)
                if specLen != len(actualKey):
                    raise ValueError("Length of key specifier '%s' does not match length of first key '%s'" %
                                     (str(keySpec), str(actualKey)))
            except TypeError:
                raise ValueError("Key specifier '%s' appears not to be a sequence type, but actual keys are " %
                                 str(keySpec) + "sequences (first key: '%s')" % str(actualKey))
        else:
            if hasattr(keySpec, "__iter__"):
                raise ValueError("Key specifier '%s' appears to be a sequence type, " % str(keySpec) +
                                 "but actual keys are not (first key: '%s')" % str(actualKey))

    def get(self, key):
        """
        Returns single value locally to driver that matches the passed key

        If multiple records are found with keys matching the passed key, a sequence of all matching
        values will be returned. If no records found, will return None
        """
        firstKey = self.first()[0]
        Data.__getKeyTypeCheck(firstKey, key)
        filteredVals = self.rdd.filter(lambda (k, v): k == key).values().collect()
        if len(filteredVals) == 1:
            return filteredVals[0]
        elif not filteredVals:
            return None
        else:
            return filteredVals

    def getMany(self, keys):
        """
        Returns values locally to driver that correspond to the passed sequence of keys.

        The return value will be a sequence equal in length to the passed keys, with each
        value in the returned sequence corresponding to the key at the same position in the passed
        keys sequence. If no value is found for a given key, the corresponding sequence element will be None.
        If multiple values are found, a sequence containing all matching values will be returned.
        """
        firstKey = self.first()[0]
        for key in keys:
            Data.__getKeyTypeCheck(firstKey, key)
        keySet = frozenset(keys)
        filteredRecs = self.rdd.filter(lambda (k, _): k in keySet).collect()
        sortingDict = {}
        for k, v in filteredRecs:
            sortingDict.setdefault(k, []).append(v)
        retVals = []
        for k in keys:
            vals = sortingDict.get(k)
            if vals is not None:
                if len(vals) == 1:
                    vals = vals[0]
            retVals.append(vals)
        return retVals

    def getRange(self, sliceOrSlices, keys=True):
        """
        Returns key/value pairs locally to driver that fall within a given range.

        The return values will be a sorted list of key/value pairs of all records in the underlying
        RDD for which the key falls within the range given by the passed slice selectors. Note that
        this may be very large, and could potentially exhaust the available memory on the driver.

        For singleton keys, a single slice (or slice sequence of length one) should be passed.
        For tuple keys, a sequence of multiple slices should be passed. A `step` attribute on slices
        is not supported and a alueError will be raised if passed.

        Parameters
        ----------
        sliceOrSlices: slice object or sequence of slices
            The passed slice or slices should be of the same cardinality as the keys of the underlying rdd.

        keys: boolean, optional, default = True
            Whether to return keys along with values, if false will just return an array of values.

        Returns
        -------
        sorted sequence of key/value pairs
        """
        # None is less than everything except itself
        def singleSlicePredicate(kv):
            key, _ = kv
            if isinstance(sliceOrSlices, slice):
                if sliceOrSlices.stop is None:
                    return key >= sliceOrSlices.start
                return sliceOrSlices.stop > key >= sliceOrSlices.start
            else:  # apparently this isn't a slice
                return key == sliceOrSlices

        def multiSlicesPredicate(kv):
            key, _ = kv
            for slise, subkey in zip(sliceOrSlices, key):
                if isinstance(slise, slice):
                    if slise.stop is None:
                        if subkey < slise.start:
                            return False
                    elif not (slise.stop > subkey >= slise.start):
                        return False
                else:  # not a slice
                    if subkey != slise:
                        return False
            return True

        firstKey = self.first()[0]
        Data.__getKeyTypeCheck(firstKey, sliceOrSlices)
        if not hasattr(sliceOrSlices, '__iter__'):
            # make my func the pFunc; http://en.wikipedia.org/wiki/P._Funk_%28Wants_to_Get_Funked_Up%29
            pFunc = singleSlicePredicate
            if hasattr(sliceOrSlices, 'step') and sliceOrSlices.step is not None:
                raise ValueError("'step' slice attribute is not supported in getRange, got step: %d" %
                                 sliceOrSlices.step)
        else:
            pFunc = multiSlicesPredicate
            for slise in sliceOrSlices:
                if hasattr(slise, 'step') and slise.step is not None:
                    raise ValueError("'step' slice attribute is not supported in getRange, got step: %d" %
                                     slise.step)

        filteredRecs = self.rdd.filter(pFunc).collect()
        # default sort of tuples is by first item, which happens to be what we want
        output = sorted(filteredRecs)

        if keys is True:
            return output
        else:
            return map(lambda (k, v): v, output)

    def __getitem__(self, item):
        # should raise exception here when no matching items found
        # see object.__getitem__ in https://docs.python.org/2/reference/datamodel.html
        isRangeQuery = False
        if isinstance(item, slice):
            isRangeQuery = True
        elif hasattr(item, '__iter__'):
            if any([isinstance(slise, slice) for slise in item]):
                isRangeQuery = True

        result = self.getRange(item, keys=False) if isRangeQuery else self.get(item)
        if (result is None) or (result == []):
            raise KeyError("No value found for key: %s" % str(item))
        return asarray(result)

    def values(self):
        """
        Return rdd of values, ignoring keys

        This calls the Spark values() method on the underlying RDD.
        """
        return self.rdd.values()

    def keys(self):
        """
        Return rdd of keys, ignoring values

        This calls the Spark keys() method on the underlying RDD.
        """
        return self.rdd.keys()

    def keysToIndices(self):
        """
        Replace the keys with one-dimensional indices from 0 to n

        Returns a new Data object
        """
        def getSize(x): yield sum(1 for _ in x)
        # First get the sizes of the partitions
        counts = self.rdd.mapPartitions(getSize).collect()
        # Get the starting indices of each partition
        counts.insert(0, 0)
        starts = self.rdd.context.broadcast(cumsum(counts)[0:-1])
        # Replace the keys
        def replaceKeys(index, iterator):
          i = starts.value[index]
          for _, x in iterator:
            yield i, x
            i += 1
        return self.replace(self.rdd.mapPartitionsWithIndex(replaceKeys), True, True)

    def astype(self, dtype, casting='safe'):
        """
        Cast values to specified numpy dtype.

        If 'smallfloat' is passed, values will be cast to the smallest floating point representation
        to which they can be cast safely, as determined by the thunder.utils.common smallest_float_type function.
        Typically this will be a float type larger than a passed integer type (for instance, float16 for int8 or uint8).

        If the passed dtype is the same as the current dtype, or if 'smallfloat' is passed when values are already
        in floating point, then this method will return self unchanged.

        Parameters
        ----------
        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast. Will return without cast if None is passed.

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Casting method to pass on to numpy's astype() method; see numpy documentation for details.

        Returns
        -------
        New Data object, of same type as self, with values cast to the requested dtype; or self if no cast is performed.
        """
        if dtype is None or dtype == '':
            return self
        from numpy import ndarray
        from numpy import dtype as dtypeFunc
        if dtype == 'smallfloat':
            # get the smallest floating point type that can be safely cast to from our current type
            from thunder.utils.common import smallestFloatType
            dtype = smallestFloatType(self.dtype)

        def cast(v, dtype_, casting_):
            if isinstance(v, ndarray):
                return v.astype(dtype_, casting=casting_, copy=False)
            else:
                # assume we are a scalar, either a numpy scalar or a python scalar
                # turn ourself into a numpy scalar of the appropriate type
                return asarray([v]).astype(dtype_, casting=casting_, copy=False)[0]

        nextRdd = self.rdd.mapValues(lambda v: cast(v, dtypeFunc(dtype), casting))
        return self._constructor(nextRdd, dtype=str(dtype)).__finalize__(self)

    def apply(self, func, keepDtype=False, keepIndex=False):
        """
        Apply arbitrary function to records of a Data object.

        This wraps the combined process of calling Spark's map operation on
        the underlying RDD and returning a reconstructed Data object.

        Parameters
        ----------
        func : function
            Function to apply to records.

        keepDtype : boolean
            Whether to preserve the dtype, if false dtype will be set to none
            under the assumption that the function might change it

        keepIndex : boolean
            Whether to preserve the index, if false index will be set to none
            under the assumption that the function might change it
        """
        return self.replace(self.rdd.map(func), keepDtype, keepIndex)

    def applyKeys(self, func, **kwargs):
        """
        Apply arbitrary function to the keys of a Data object, preserving the values.

        See also
        --------
        Data.apply
        """

        return self.apply(lambda (k, v): (func(k), v), **kwargs)

    def applyValues(self, func, **kwargs):
        """
        Apply arbitrary function to the values of a Data object, preserving the keys.

        See also
        --------
        Data.apply
        """
        return self.apply(lambda (k, v): (k, func(v)), **kwargs)

    def collect(self, sorting=False):
        """
        Return all records locally to the driver

        This will be slow for large datasets, and may exhaust the available memory on the driver.

        This calls the Spark collect() method on the underlying RDD.
        """
        if sorting:
            return self.sortByKey().rdd.collect()
        else:
            return self.rdd.collect()

    def collectAsArray(self, sorting=False):
        """
        Return all keys and values locally to the driver as a tuple of numpy arrays

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        out = self.collect(sorting)
        keys = asarray(map(lambda (k, v): k, out))
        values = asarray(map(lambda (k, v): v, out))
        return keys, values

    def collectValuesAsArray(self, sorting=False):
        """
        Return all records to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        if sorting:
            rdd = self.sortByKey().rdd
        else:
            rdd = self.rdd
        return asarray(rdd.values().collect())

    def collectKeysAsArray(self, sorting=False):
        """
        Return all values locally to the driver as a numpy array

        This will be slow for large datasets, and may exhaust the available memory on the driver.
        """
        if sorting:
            rdd = self.sortByKey().rdd
        else:
            rdd = self.rdd
        return asarray(rdd.keys().collect())

    def sortByKey(self):
        """
        Sort records by keys.

        This calls the Spark sortByKey() method on the underlying RDD, but reverse the order
        of the key tuples before and after sorting so they are sorted according to the convention
        that the first key varies fastest, then the second, then the third, etc.
        """
        newrdd = self.rdd.map(lambda (k, v): (k[::-1], v)).sortByKey().map(lambda (k, v): (k[::-1], v))
        return self._constructor(newrdd).__finalize__(self)

    def count(self):
        """
        Calculates and returns the number of records in the RDD.

        This calls the Spark count() method on the underlying RDD and updates
        the .nrecords metadata attribute.
        """
        count = self.rdd.count()
        self._nrecords = count
        return count

    def mean(self, dtype='float64', casting='safe'):
        """
        Mean of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested
        type before the operation is performed. See Data.astype() for details.
        """
        return self.stats('mean', dtype=dtype, casting=casting).mean()

    def sum(self, dtype='float64', casting='safe'):
        """
        Sum of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.sum() is equivalent to obj.astype(dtype, casting).rdd.values().sum().
        """
        out = self.astype(dtype, casting)
        return out.rdd.values().sum()

    def variance(self, dtype='float64', casting='safe'):
        """
        Variance of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.
        """
        return self.stats('variance', dtype=dtype, casting=casting).variance()

    def stdev(self, dtype='float64', casting='safe'):
        """
        Standard deviation of values computed by aggregating across records, returned as an ndarray
        with the same size as a single record.

        If dtype is not None, then the values will first be cast to the requested type before the operation is
        performed. See Data.astype() for details.

        obj.stdev() is equivalent to obj.astype(dtype, casting).rdd.values().stdev().
        """
        return self.stats('stdev', dtype=dtype, casting=casting).stdev()

    def stats(self, requestedStats='all', dtype='float64', casting='safe'):
        """
        Return a L{StatCounter} object that captures all or some of the mean, variance, maximum, minimum,
        and count of the RDD's elements in one operation.

        If dtype is specified and not None, will first cast the data as described in Data.astype().

        Parameters
        ----------
        requestedStats: sequence of one or more requested stats, or 'all'
            Possible stats include 'mean', 'sum', 'min', 'max', 'variance', 'sampleVariance', 'stdev', 'sampleStdev'.

        dtype: numpy dtype or dtype specifier, or string 'smallfloat', or None
            Data type to which RDD values are to be cast before calculating stats. See Data.astype().

        casting: 'no'|'equiv'|'safe'|'same_kind'|'unsafe', optional, default 'safe'
            Method of casting to use. See Data.astype() and numpy astype() function.
        """
        from thunder.utils.statcounter import StatCounter

        def redFunc(left_counter, right_counter):
            return left_counter.mergeStats(right_counter)

        out = self.astype(dtype, casting)
        return out.values().mapPartitions(lambda i: [StatCounter(i, stats=requestedStats)]).reduce(redFunc)

    def max(self):
        """ Maximum of values across keys, returned as an ndarray. """
        # NOTE: Does not use stats('max') to prevent cast to float64
        return self.rdd.values().reduce(maximum)

    def min(self):
        """ Minimum of values across keys, returned as an ndarray. """
        # NOTE: Does not use stats('min') to prevent cast to float64
        return self.rdd.values().reduce(minimum)

    def coalesce(self, numPartitions):
        """
        Coalesce data (used to reduce number of partitions).

        This calls the Spark coalesce() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in coalesced RDD
        """
        current = self.rdd.getNumPartitions()
        if numPartitions > current:
            raise Exception('Trying to increase number of partitions (from %g to %g), '
                            'cannot use coalesce, try repartition' % (current, numPartitions))
        self.rdd = self.rdd.coalesce(numPartitions)
        return self

    def cache(self):
        """
        Enable in-memory caching.

        This calls the Spark cache() method on the underlying RDD.
        """
        self.rdd.cache()
        return self

    def repartition(self, numPartitions):
        """
        Repartition data.

        This calls the Spark repartition() method on the underlying RDD.

        Parameters
        ----------
        numPartitions : int
            Number of partitions in new RDD
        """
        self.rdd = self.rdd.repartition(numPartitions)
        return self

    def replace(self, newRdd, keepDtype=False, keepIndex=False):
        """
        Replace the rdd inside a Data object.

        This wraps function handles replacing the underlying RDD and
        returning a reconstructed Data object.

        Parameters
        ----------
        func : function
            Function to apply to records.

        keepDtype : boolean
            Whether to preserve the dtype, if false dtype will be set to none
            under the assumption that the function might change it

        keepIndex : boolean
            Whether to preserve the index, if false index will be set to none
            under the assumption that the function might change it
        """
        noprop = ()
        if keepDtype is False:
            noprop += ('_dtype',)
        if keepIndex is False:
            noprop += ('_index',)
        return self._constructor(newRdd).__finalize__(self, noPropagate=noprop)

    def filter(self, func):
        """
        Filter records by applying a function to each record.

        This calls the Spark filter() method on the underlying RDD.

        Parameters
        ----------
        func : function
            The function to compute on each record, should evaluate to Boolean
        """
        return self._constructor(self.rdd.filter(lambda d: func(d))).__finalize__(self)._resetCounts()

    def filterOnKeys(self, func):
        """
        Filter records by applying a function to keys

        See also
        --------
        Data.filter
        """
        return self._constructor(self.rdd.filter(lambda (k, v): func(k))).__finalize__(self)._resetCounts()

    def filterOnValues(self, func):
        """
        Filter records by applying a function to values

        See also
        --------
        Data.filter
        """
        return self._constructor(self.rdd.filter(lambda (k, v): func(v))).__finalize__(self)._resetCounts()

    def range(self, start, stop):
        """
        Extract records with keys that fall inside a range

        Returns another Data object, unlike
        getRange which returns a local array to the driver

        See also
        --------
        Data.filterOnKeys
        """
        if start >= stop:
            raise ValueError("Start (%g) is greater than or equal to stop (%g), result will be empty" % (start, stop))

        return self.filterOnKeys(lambda k: start <= k < stop)