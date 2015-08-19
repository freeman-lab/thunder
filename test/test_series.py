from numpy import allclose, amax, arange, array, array_equal
from numpy import dtype as dtypeFunc
from numpy.testing import assert_array_equal, assert_equal
try:
    from nose.tools import assert_equals, assert_is_none, assert_is_not_none, assert_raises, assert_true
except:
    from nose_extra_tools import assert_equals, assert_is_none, assert_is_not_none, assert_raises, assert_true

from thunder.rdds.series import Series
from test_utils import *


class TestSeriesConversions(PySparkTestCase):

    def test_toRowMatrix(self):
        from thunder.rdds.matrices import RowMatrix
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        mat = data.toRowMatrix()
        assert(isinstance(mat, RowMatrix))
        assert(mat.nrows == 2)
        assert(mat.ncols == 4)

        # check a basic operation from superclass
        newmat = mat.applyValues(lambda x: x + 1)
        out = newmat.collectValuesAsArray()
        assert(array_equal(out, array([[5, 6, 7, 8], [9, 10, 11, 12]])))

    def test_toTimeSeries(self):
        from thunder.rdds.timeseries import TimeSeries
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd)
        ts = data.toTimeSeries()
        assert(isinstance(ts, TimeSeries))

    def test_toImages(self):
        from thunder.rdds.images import Images
        rdd = self.sc.parallelize([((0, 0), array([1])), ((0, 1), array([2])),
                                   ((1, 0), array([3])), ((1, 1), array([4]))])
        data = Series(rdd)
        imgs = data.toImages()
        assert(isinstance(imgs, Images))

        im = imgs.values().first()
        assert(allclose(im, [[1, 2], [3, 4]]))

    def test_castToFloat(self):
        from numpy import arange
        shape = (3, 2, 2)
        size = 3*2*2
        ary = arange(size, dtype=dtypeFunc('uint8')).reshape(shape)
        ary2 = ary + size
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        series = SeriesLoader(self.sc).fromArraysAsImages([ary, ary2])

        castSeries = series.astype("smallfloat")

        assert_equals('float16', str(castSeries.dtype))
        assert_equals('float16', str(castSeries.first()[1].dtype))


class TestSeriesDataStatsMethods(PySparkTestCase):
    def generateTestSeries(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary1 = arange(8, dtype=dtypeFunc('uint8')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('uint8')).reshape((2, 4))
        return SeriesLoader(self.sc).fromArraysAsImages([ary1, ary2])

    def test_mean(self):
        from test_utils import elementwiseMean
        series = self.generateTestSeries()
        meanVal = series.mean()

        expected = elementwiseMean(series.values().collect())
        assert_true(allclose(expected, meanVal))
        assert_equals('float64', str(meanVal.dtype))

    def test_sum(self):
        from numpy import add
        series = self.generateTestSeries()
        sumVal = series.sum(dtype='float32')

        arys = series.values().collect()
        expected = reduce(add, arys)
        assert_true(array_equal(expected, sumVal))
        assert_equals('float32', str(sumVal.dtype))

    def test_variance(self):
        from test_utils import elementwiseVar
        series = self.generateTestSeries()
        varVal = series.variance()

        arys = series.values().collect()
        expected = elementwiseVar([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, varVal))
        assert_equals('float64', str(varVal.dtype))

    def test_stdev(self):
        from test_utils import elementwiseStdev
        series = self.generateTestSeries()
        stdVal = series.stdev()

        arys = series.values().collect()
        expected = elementwiseStdev([ary.astype('float16') for ary in arys])
        assert_true(allclose(expected, stdVal, atol=0.001))
        assert_equals('float64', str(stdVal.dtype))  # why not float16? see equivalent Images test

    def test_stats(self):
        from test_utils import elementwiseMean, elementwiseVar
        series = self.generateTestSeries()
        statsVal = series.stats()

        arys = series.values().collect()
        floatArys = [ary.astype('float16') for ary in arys]
        expectedMean = elementwiseMean(floatArys)
        expectedVar = elementwiseVar(floatArys)
        assert_true(allclose(expectedMean, statsVal.mean()))
        assert_true(allclose(expectedVar, statsVal.variance()))

    def test_max(self):
        from numpy import maximum
        series = self.generateTestSeries()
        maxVal = series.max()
        arys = series.values().collect()
        assert_true(array_equal(reduce(maximum, arys), maxVal))

    def test_min(self):
        from numpy import minimum
        series = self.generateTestSeries()
        minVal = series.min()
        arys = series.values().collect()
        assert_true(array_equal(reduce(minimum, arys), minVal))


class TestSeriesMethods(PySparkTestCase):

    def test_between(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd).between(0, 1)
        assert(allclose(data.index, array([0, 1])))
        assert(allclose(data.first()[1], array([4, 5])))

    def test_select(self):
        rdd = self.sc.parallelize([(0, array([4, 5, 6, 7])), (1, array([8, 9, 10, 11]))])
        data = Series(rdd, index=['label1', 'label2', 'label3', 'label4'])
        selection1 = data.select(['label1'])
        assert(allclose(selection1.first()[1], 4))
        selection1 = data.select('label1')
        assert(allclose(selection1.first()[1], 4))
        selection2 = data.select(['label1', 'label2'])
        assert(allclose(selection2.first()[1], array([4, 5])))

    def test_seriesStats(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5]))])
        data = Series(rdd)
        assert(allclose(data.seriesMean().first()[1], 3.0))
        assert(allclose(data.seriesSum().first()[1], 15.0))
        assert(allclose(data.seriesMedian().first()[1], 3.0))
        assert(allclose(data.seriesStdev().first()[1], 1.4142135))
        assert(allclose(data.seriesStat('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('mean').first()[1], 3.0))
        assert(allclose(data.seriesStats().select('count').first()[1], 5))
        assert(allclose(data.seriesPercentile(25).first()[1], 2.0))
        assert(allclose(data.seriesPercentile((25, 75)).first()[1], array([2.0, 4.0])))

    def test_standardization_axis0(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        centered = data.center(0)
        standardized = data.standardize(0)
        zscored = data.zscore(0)
        assert(allclose(centered.first()[1], array([-2, -1, 0, 1, 2]), atol=1e-3))
        assert(allclose(standardized.first()[1], array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553]), atol=1e-3))
        assert(allclose(zscored.first()[1], array([-1.41421, -0.70710,  0,  0.70710,  1.41421]), atol=1e-3))

    def test_standardization_axis1(self):
        rdd = self.sc.parallelize([(0, array([1, 2], dtype='float16')), (0, array([3, 4], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        centered = data.center(1)
        standardized = data.standardize(1)
        zscored = data.zscore(1)
        assert(allclose(centered.first()[1], array([-1, -1]), atol=1e-3))
        assert(allclose(standardized.first()[1], array([1, 2]), atol=1e-3))
        assert(allclose(zscored.first()[1], array([-1, -1]), atol=1e-3))

    def test_squelch(self):
        rdd = self.sc.parallelize([(0, array([1, 2])), (0, array([3, 4]))])
        data = Series(rdd)
        squelched = data.squelch(5)
        assert(allclose(squelched.collectValuesAsArray(), [[0, 0], [0, 0]]))
        squelched = data.squelch(3)
        assert(allclose(squelched.collectValuesAsArray(), [[0, 0], [3, 4]]))
        squelched = data.squelch(1)
        assert(allclose(squelched.collectValuesAsArray(), [[1, 2], [3, 4]]))

    def test_correlate(self):
        rdd = self.sc.parallelize([(0, array([1, 2, 3, 4, 5], dtype='float16'))])
        data = Series(rdd, dtype='float16')
        sig1 = [4, 5, 6, 7, 8]
        corrData = data.correlate(sig1)
        assert_equals('float64', corrData._dtype)
        corr = corrData.values().collect()
        assert(allclose(corr[0], 1))
        sig12 = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
        corrs = data.correlate(sig12).values().collect()
        assert(allclose(corrs[0], [1, -1]))

    def test_subset(self):
        rdd = self.sc.parallelize([(0, array([1, 5], dtype='float16')),
                                   (0, array([1, 10], dtype='float16')),
                                   (0, array([1, 15], dtype='float16'))])
        data = Series(rdd)
        assert_equal(len(data.subset(3, stat='min', thresh=0)), 3)
        assert_array_equal(data.subset(1, stat='max', thresh=10), [[1, 15]])
        assert_array_equal(data.subset(1, stat='mean', thresh=6), [[1, 15]])
        assert_array_equal(data.subset(1, stat='std', thresh=6), [[1, 15]])
        assert_array_equal(data.subset(1, thresh=6), [[1, 15]])

    def test_query_subscripts(self):
        dataLocal = [
            ((1, 1), array([1.0, 2.0, 3.0])),
            ((2, 1), array([2.0, 2.0, 4.0])),
            ((1, 2), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear(self):
        dataLocal = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2]), array([3])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert(allclose(values[1, :], array([4.0, 2.0, 1.0])))

    def test_query_linear_singleton(self):
        dataLocal = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]

        data = Series(self.sc.parallelize(dataLocal))

        inds = array([array([1, 2])])
        keys, values = data.query(inds)
        assert(allclose(values[0, :], array([1.5, 2., 3.5])))
        assert_equals(data.dtype, values[0, :].dtype)

    def test_maxProject(self):
        from thunder.rdds.fileio.seriesloader import SeriesLoader
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArraysAsImages(ary)
        project0Series = series.maxProject(axis=0)
        project0 = project0Series.pack()

        project1Series = series.maxProject(axis=1)
        project1 = project1Series.pack(sorting=True)

        assert_true(array_equal(amax(ary.T, 0), project0))
        assert_true(array_equal(amax(ary.T, 1), project1))

    def test_index_setter_getter(self):
        dataLocal = [
            ((1,), array([1.0, 2.0, 3.0])),
            ((2,), array([2.0, 2.0, 4.0])),
            ((3,), array([4.0, 2.0, 1.0]))
        ]
        data = Series(self.sc.parallelize(dataLocal))

        assert_true(array_equal(data.index, array([0, 1, 2])))
        data.index = [3, 2, 1]
        assert_true(data.index == [3, 2, 1])

        def setIndex(data, idx):
            data.index = idx

        assert_raises(ValueError, setIndex, data, 5)
        assert_raises(ValueError, setIndex, data, [1, 2])

    def test_selectByIndex(self):
        dataLocal = [((1,), arange(12))]
        index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        data = Series(self.sc.parallelize(dataLocal), index=index)

        result = data.selectByIndex(1)
        assert_true(array_equal(result.values().first(), array([4, 5, 6, 7])))
        assert_true(array_equal(result.index, array([1, 1, 1, 1])))

        result = data.selectByIndex(1, squeeze=True)
        assert_true(array_equal(result.index, array([0, 1, 2, 3])))

        index = [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
        ]
        data.index = array(index).T

        result, mask = data.selectByIndex(0, level=2, returnMask=True)
        assert_true(array_equal(result.values().first(), array([0, 2, 6, 8])))
        assert_true(array_equal(result.index, array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])))
        assert_true(array_equal(mask, array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])))

        result = data.selectByIndex(0, level=2, squeeze=True)
        assert_true(array_equal(result.values().first(), array([0, 2, 6, 8])))
        assert_true(array_equal(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]])))

        result = data.selectByIndex([1, 0], level=[0, 1])
        assert_true(array_equal(result.values().first(), array([6, 7])))
        assert_true(array_equal(result.index, array([[1, 0, 0], [1, 0, 1]])))

        result = data.selectByIndex(val=[0, [2,3]], level=[0, 2])
        assert_true(array_equal(result.values().first(), array([4, 5])))
        assert_true(array_equal(result.index, array([[0, 1, 2], [0, 1, 3]])))

        result = data.selectByIndex(1, level=1, filter=True)
        assert_true(array_equal(result.values().first(), array([0, 1, 6, 7])))
        assert_true(array_equal(result.index, array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])))

    def test_seriesAggregateByIndex(self):
        dataLocal = [((1,), arange(12))]
        index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        data = Series(self.sc.parallelize(dataLocal), index=index)

        result = data.seriesAggregateByIndex(sum)
        print result.values().first()
        assert_true(array_equal(result.values().first(), array([6, 22, 38])))
        assert_true(array_equal(result.index, array([0, 1, 2])))

        index = [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
        ]
        data.index = array(index).T
        
        result = data.seriesAggregateByIndex(sum, level=[0, 1])
        assert_true(array_equal(result.values().first(), array([1, 14, 13, 38])))
        assert_true(array_equal(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]])))

    def test_seriesStatByIndex(self):
        dataLocal = [((1,), arange(12))]
        index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        data = Series(self.sc.parallelize(dataLocal), index=index)

        assert_true(array_equal(data.seriesStatByIndex('sum').values().first(), array([6, 22, 38])))
        assert_true(array_equal(data.seriesStatByIndex('mean').values().first(), array([1.5, 5.5, 9.5])))
        assert_true(array_equal(data.seriesStatByIndex('min').values().first(), array([0, 4, 8])))
        assert_true(array_equal(data.seriesStatByIndex('max').values().first(), array([3, 7, 11])))
        assert_true(array_equal(data.seriesStatByIndex('count').values().first(), array([4, 4, 4])))
        assert_true(array_equal(data.seriesStatByIndex('median').values().first(), array([1.5, 5.5, 9.5])))

        assert_true(array_equal(data.seriesSumByIndex().values().first(), array([6, 22, 38])))
        assert_true(array_equal(data.seriesMeanByIndex().values().first(), array([1.5, 5.5, 9.5])))
        assert_true(array_equal(data.seriesMinByIndex().values().first(), array([0, 4, 8])))
        assert_true(array_equal(data.seriesMaxByIndex().values().first(), array([3, 7, 11])))
        assert_true(array_equal(data.seriesCountByIndex().values().first(), array([4, 4, 4])))
        assert_true(array_equal(data.seriesMedianByIndex().values().first(), array([1.5, 5.5, 9.5])))

        index = [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
        ]
        data.index = array(index).T

        result = data.seriesStatByIndex('sum', level=[0, 1])
        assert_true(array_equal(result.values().first(), array([1, 14, 13, 38])))
        assert_true(array_equal(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]])))

        result = data.seriesSumByIndex(level=[0, 1])
        assert_true(array_equal(result.values().first(), array([1, 14, 13, 38])))
        assert_true(array_equal(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]])))

    def test_groupByFixedLength(self):
        rdd = self.sc.parallelize([((0,), array([0, 1, 2, 3, 4, 5, 6, 7], dtype='float16'))])
        data = Series(rdd)

        test1 = data.groupByFixedLength(4)
        assert(test1.keys().collect() == [(0, 0), (0, 1)])
        assert(allclose(test1.index, array([0, 1, 2, 3])))
        assert(allclose(test1.values().collect(), [[0, 1, 2, 3], [4, 5, 6, 7]]))

        test2 = data.groupByFixedLength(2)
        assert(test2.keys().collect() == [(0, 0), (0, 1), (0, 2), (0, 3)])
        assert(allclose(test2.index, array([0, 1])))
        assert(allclose(test2.values().collect(), [[0, 1], [2, 3], [4, 5], [6, 7]]))

    def test_meanByFixedLength(self):
        rdd = self.sc.parallelize([((0,), array([0, 1, 2, 3, 4, 5, 6, 7], dtype='float16'))])
        data = Series(rdd)

        test1 = data.meanByFixedLength(4)
        assert(test1.keys().collect() == [(0,)])
        assert(allclose(test1.index, array([0, 1, 2, 3])))
        assert(allclose(test1.values().collect(), [[2, 3, 4, 5]]))

        test2 = data.meanByFixedLength(2)
        assert(test2.keys().collect() == [(0,)])
        assert(allclose(test2.index, array([0, 1])))
        assert(allclose(test2.values().collect(), [[3, 4]]))


class TestSeriesRegionMeanMethods(PySparkTestCase):
    def setUp(self):
        super(TestSeriesRegionMeanMethods, self).setUp()
        self.dataLocal = [
            ((0, 0), array([1.0, 2.0, 3.0])),
            ((0, 1), array([2.0, 2.0, 4.0])),
            ((1, 0), array([4.0, 2.0, 1.0])),
            ((1, 1), array([3.0, 1.0, 1.0]))
        ]
        self.series = Series(self.sc.parallelize(self.dataLocal),
                             dtype=self.dataLocal[0][1].dtype,
                             index=arange(3))

    def __setup_meanByRegion(self, useMask=False):
        itemIdxs = [1, 2]  # data keys for items 1 and 2 (0-based)
        keys = [self.dataLocal[idx][0] for idx in itemIdxs]

        expectedKeys = tuple(vstack(keys).mean(axis=0).astype('int16'))
        expected = vstack([self.dataLocal[idx][1] for idx in itemIdxs]).mean(axis=0)
        if useMask:
            keys = array([[0, 1], [1, 0]], dtype='uint8')
        return keys, expectedKeys, expected

    @staticmethod
    def __checkAsserts(expectedLen, expectedKeys, expected, actual):
        assert_equals(expectedLen, len(actual))
        assert_equals(expectedKeys, actual[0])
        assert_true(array_equal(expected, actual[1]))

    @staticmethod
    def __checkNestedAsserts(expectedLen, expectedKeys, expected, actual):
        assert_equals(expectedLen, len(actual))
        for i in xrange(expectedLen):
            assert_equals(expectedKeys[i], actual[i][0])
            assert_true(array_equal(expected[i], actual[i][1]))

    def __checkReturnedSeriesAttributes(self, newSeries):
        assert_true(newSeries._dims is None)  # check that new _dims is unset
        assert_equals(self.series.dtype, newSeries._dtype)  # check that new dtype is set
        assert_true(array_equal(self.series.index, newSeries._index))  # check that new index is set
        assert_is_not_none(newSeries.dims)  # check that new dims is at least calculable (expected to be meaningless)

    def __run_tst_meanOfRegion(self, useMask):
        keys, expectedKeys, expected = self.__setup_meanByRegion(useMask)
        actual = self.series.meanOfRegion(keys)
        TestSeriesRegionMeanMethods.__checkAsserts(2, expectedKeys, expected, actual)

    def test_meanOfRegion(self):
        self.__run_tst_meanOfRegion(False)

    def test_meanOfRegionWithMask(self):
        self.__run_tst_meanOfRegion(True)

    def test_meanOfRegionErrorsOnMissing(self):
        _, expectedKeys, expected = self.__setup_meanByRegion(False)
        keys = [(17, 24), (17, 25)]
        # if no records match, return None, None
        actualKey, actualVal = self.series.meanOfRegion(keys)
        assert_is_none(actualKey)
        assert_is_none(actualVal)
        # if we have only a partial match but haven't turned on validation, return a sensible value
        keys = [(0, 1), (17, 25)]
        actualKey, actualVal = self.series.meanOfRegion(keys)
        assert_equals((0, 1), actualKey)
        assert_true(array_equal(self.dataLocal[1][1], actualVal))
        # throw an error on a partial match when validation turned on
        assert_raises(ValueError, self.series.meanOfRegion, keys, validate=True)

    def test_meanByRegions_singleRegion(self):
        keys, expectedKeys, expected = self.__setup_meanByRegion()

        actualSeries = self.series.meanByRegions([keys])
        actual = actualSeries.collect()
        self.__checkReturnedSeriesAttributes(actualSeries)
        TestSeriesRegionMeanMethods.__checkNestedAsserts(1, [expectedKeys], [expected], actual)

    def test_meanByRegionsErrorsOnMissing(self):
        keys, expectedKeys, expected = self.__setup_meanByRegion()
        keys += [(17, 25)]

        # check that we get a sensible value with validation turned off:
        actualSeries = self.series.meanByRegions([keys])
        actual = actualSeries.collect()
        self.__checkReturnedSeriesAttributes(actualSeries)
        TestSeriesRegionMeanMethods.__checkNestedAsserts(1, [expectedKeys], [expected], actual)

        # throw an error on a partial match when validation turned on
        # this error will be on the workers, which propagates back to the driver
        # as something other than the ValueError that it started out life as
        assert_raises(Exception, self.series.meanByRegions([keys], validate=True).count)

    def test_meanByRegions_singleRegionWithMask(self):
        mask, expectedKeys, expected = self.__setup_meanByRegion(True)

        actualSeries = self.series.meanByRegions(mask)
        actual = actualSeries.collect()
        self.__checkReturnedSeriesAttributes(actualSeries)
        TestSeriesRegionMeanMethods.__checkNestedAsserts(1, [expectedKeys], [expected], actual)

    def test_meanByRegions_twoRegions(self):
        nestedKeys, expectedKeys, expected = [], [], []
        for itemIdxs in [(0, 1), (1, 2)]:
            keys = [self.dataLocal[idx][0] for idx in itemIdxs]
            nestedKeys.append(keys)
            avgKeys = tuple(vstack(keys).mean(axis=0).astype('int16'))
            expectedKeys.append(avgKeys)
            avgVals = vstack([self.dataLocal[idx][1] for idx in itemIdxs]).mean(axis=0)
            expected.append(avgVals)

        actualSeries = self.series.meanByRegions(nestedKeys)
        actual = actualSeries.collect()
        self.__checkReturnedSeriesAttributes(actualSeries)
        TestSeriesRegionMeanMethods.__checkNestedAsserts(2, expectedKeys, expected, actual)

    def test_meanByRegions_twoRegionsWithMask(self):
        expectedKeys, expected = [], []
        mask = array([[1, 1], [2, 0]], dtype='uint8')
        for itemIdxs in [(0, 1), (2, )]:
            keys = [self.dataLocal[idx][0] for idx in itemIdxs]
            avgKeys = tuple(vstack(keys).mean(axis=0).astype('int16'))
            expectedKeys.append(avgKeys)
            avgVals = vstack([self.dataLocal[idx][1] for idx in itemIdxs]).mean(axis=0)
            expected.append(avgVals)

        actualSeries = self.series.meanByRegions(mask)
        actual = actualSeries.collect()
        self.__checkReturnedSeriesAttributes(actualSeries)
        TestSeriesRegionMeanMethods.__checkNestedAsserts(2, expectedKeys, expected, actual)

