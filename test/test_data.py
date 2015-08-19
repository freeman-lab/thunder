try:
    from nose.tools import assert_equals, assert_is_none, assert_is_not_none, assert_raises, assert_true
except:
    from nose_extra_tools import assert_equals, assert_is_none, assert_is_not_none, assert_raises, assert_true
from numpy import allclose, array, array_equal, can_cast, float32

from thunder.rdds.data import Data
from test_utils import PySparkTestCase


class TestImagesGetters(PySparkTestCase):
    """Test `get` and related methods on an Images-like Data object
    """
    def setUp(self):
        super(TestImagesGetters, self).setUp()
        self.ary1 = array([[1, 2], [3, 4]], dtype='int16')
        self.ary2 = array([[5, 6], [7, 8]], dtype='int16')
        rdd = self.sc.parallelize([(0, self.ary1), (1, self.ary2)])
        self.images = Data(rdd, dtype='int16')

    def test_getMissing(self):
        assert_is_none(self.images.get(-1))

    def test_get(self):
        assert_true(array_equal(self.ary2, self.images.get(1)))

        # keys are integers, ask for sequence
        assert_raises(ValueError, self.images.get, (1, 2))

    def test_getMany(self):
        vals = self.images.getMany([0, -1, 1, 0])
        assert_equals(4, len(vals))
        assert_true(array_equal(self.ary1, vals[0]))
        assert_is_none(vals[1])
        assert_true(array_equal(self.ary2, vals[2]))
        assert_true(array_equal(self.ary1, vals[3]))

        # keys are integers, ask for sequences:
        assert_raises(ValueError, self.images.get, [(0, 0)])
        assert_raises(ValueError, self.images.get, [0, (0, 0), 1, 0])

    def test_getRanges(self):
        vals = self.images.getRange(slice(None))
        assert_equals(2, len(vals))
        assert_equals(0, vals[0][0])
        assert_equals(1, vals[1][0])
        assert_true(array_equal(self.ary1, vals[0][1]))
        assert_true(array_equal(self.ary2, vals[1][1]))

        vals = self.images.getRange(slice(0, 1))
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        vals = self.images.getRange(slice(1))
        assert_equals(1, len(vals))
        assert_equals(0, vals[0][0])
        assert_true(array_equal(self.ary1, vals[0][1]))

        vals = self.images.getRange(slice(1, 2))
        assert_equals(1, len(vals))
        assert_equals(1, vals[0][0])
        assert_true(array_equal(self.ary2, vals[0][1]))

        vals = self.images.getRange(slice(2, 3))
        assert_equals(0, len(vals))

        # keys are integers, ask for sequence
        assert_raises(ValueError, self.images.getRange, [slice(1), slice(1)])

        # raise exception if 'step' specified:
        assert_raises(ValueError, self.images.getRange, slice(1, 2, 2))

    def test_brackets(self):
        vals = self.images[1]
        assert_true(array_equal(self.ary2, vals))

        vals = self.images[0:1]
        assert_equals(1, len(vals))
        assert_true(array_equal(self.ary1, vals[0]))

        vals = self.images[:]
        assert_equals(2, len(vals))
        assert_true(array_equal(self.ary1, vals[0]))
        assert_true(array_equal(self.ary2, vals[1]))

        vals = self.images[1:4]
        assert_equals(1, len(vals))
        assert_true(array_equal(self.ary2, vals[0]))

        vals = self.images[1:]
        assert_equals(1, len(vals))
        assert_true(array_equal(self.ary2, vals[0]))

        vals = self.images[:1]
        assert_equals(1, len(vals))
        assert_true(array_equal(self.ary1, vals[0]))

        assert_raises(KeyError, self.images.__getitem__, 2)
        assert_raises(KeyError, self.images.__getitem__, slice(2, 3))


class TestSeriesGetters(PySparkTestCase):
    """Test `get` and related methods on a Series-like Data object
    """
    def setUp(self):
        super(TestSeriesGetters, self).setUp()
        self.dataLocal = [
            ((0, 0), array([1.0, 2.0, 3.0], dtype='float32')),
            ((0, 1), array([2.0, 2.0, 4.0], dtype='float32')),
            ((1, 0), array([4.0, 2.0, 1.0], dtype='float32')),
            ((1, 1), array([3.0, 1.0, 1.0], dtype='float32'))
        ]
        self.series = Data(self.sc.parallelize(self.dataLocal), dtype='float32')

    def test_getMissing(self):
        assert_is_none(self.series.get((-1, -1)))

    def test_get(self):
        expected = self.dataLocal[1][1]
        assert_true(array_equal(expected, self.series.get((0, 1))))

        assert_raises(ValueError, self.series.get, 1)  # keys are sequences, ask for integer
        assert_raises(ValueError, self.series.get, (1, 2, 3))  # key length mismatch

    def test_getMany(self):
        vals = self.series.getMany([(0, 0), (17, 256), (1, 0), (0, 0)])
        assert_equals(4, len(vals))
        assert_true(array_equal(self.dataLocal[0][1], vals[0]))
        assert_is_none(vals[1])
        assert_true(array_equal(self.dataLocal[2][1], vals[2]))
        assert_true(array_equal(self.dataLocal[0][1], vals[3]))

        assert_raises(ValueError, self.series.getMany, [1])  # keys are sequences, ask for integer
        assert_raises(ValueError, self.series.getMany, [(0, 0), 1, (1, 0), (0, 0)])  # asking for integer again

    def test_getRanges(self):
        vals = self.series.getRange([slice(2), slice(2)])
        assert_equals(4, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_equals(self.dataLocal[2][0], vals[2][0])
        assert_equals(self.dataLocal[3][0], vals[3][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3][1]))

        vals = self.series.getRange([slice(2), slice(1)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[2][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[1][1]))

        vals = self.series.getRange([slice(None), slice(1, 2)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_equals(self.dataLocal[3][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[1][1]))

        vals = self.series.getRange([slice(None), slice(None)])
        assert_equals(4, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_equals(self.dataLocal[2][0], vals[2][0])
        assert_equals(self.dataLocal[3][0], vals[3][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2][1]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3][1]))

        vals = self.series.getRange([0, slice(None)])
        assert_equals(2, len(vals))
        assert_equals(self.dataLocal[0][0], vals[0][0])
        assert_equals(self.dataLocal[1][0], vals[1][0])
        assert_true(array_equal(self.dataLocal[0][1], vals[0][1]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1][1]))

        vals = self.series.getRange([0, 1])
        assert_equals(1, len(vals))
        assert_equals(self.dataLocal[1][0], vals[0][0])
        assert_true(array_equal(self.dataLocal[1][1], vals[0][1]))

        vals = self.series.getRange([slice(2, 3), slice(None)])
        assert_equals(0, len(vals))

        # keys are sequences, ask for single slice
        assert_raises(ValueError, self.series.getRange, slice(2, 3))

        # ask for wrong number of slices
        assert_raises(ValueError, self.series.getRange, [slice(2, 3), slice(2, 3), slice(2, 3)])

        # raise exception if 'step' specified:
        assert_raises(ValueError, self.series.getRange, [slice(0, 4, 2), slice(2, 3)])

    def test_brackets(self):
        # returns just value; calls `get`
        vals = self.series[(1, 0)]
        assert_true(array_equal(self.dataLocal[2][1], vals))

        # tuple isn't needed; returns just value, calls `get`
        vals = self.series[0, 1]
        assert_true(array_equal(self.dataLocal[1][1], vals))

        # if slices are passed, calls `getRange`, returns values
        vals = self.series[0:1, 1:2]
        assert_equals(1, len(vals))
        assert_true(array_equal(self.dataLocal[1][1], vals[0]))

        # if slice extends out of bounds, return only the elements that are in bounds
        vals = self.series[:4, :1]
        assert_equals(2, len(vals))
        assert_true(array_equal(self.dataLocal[0][1], vals[0]))
        assert_true(array_equal(self.dataLocal[2][1], vals[1]))

        # empty slice works
        vals = self.series[:, 1:2]
        assert_equals(2, len(vals))
        assert_true(array_equal(self.dataLocal[1][1], vals[0]))
        assert_true(array_equal(self.dataLocal[3][1], vals[1]))

        # multiple empty slices work
        vals = self.series[:, :]
        assert_equals(4, len(vals))
        assert_true(array_equal(self.dataLocal[0][1], vals[0]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1]))
        assert_true(array_equal(self.dataLocal[2][1], vals[2]))
        assert_true(array_equal(self.dataLocal[3][1], vals[3]))

        # mixing slices and individual indicies works:
        vals = self.series[0, :]
        assert_equals(2, len(vals))
        assert_true(array_equal(self.dataLocal[0][1], vals[0]))
        assert_true(array_equal(self.dataLocal[1][1], vals[1]))

        # trying to getitem a key that doesn't exist throws a KeyError
        assert_raises(KeyError, self.series.__getitem__, (25, 17))

        # passing a range that is completely out of bounds throws a KeyError
        assert_raises(KeyError, self.series.__getitem__, (slice(2, 3), slice(None, None)))


class TestCasting(PySparkTestCase):
    def setUp(self):
        super(TestCasting, self).setUp()
        # float16 max value is 6.55040e+04 (np.finfo(np.float16))
        # "*Big*" values are too large to cast safely down to float16s
        DATA = [
            ('float32Array', array([1.1, 2.2], dtype='float32')),
            ('float32BigArray', array([1.1e+05, 2.2e+05], dtype='float32')),
            ('float32Scalar', float32(1.1)),
            ('float32BigScalar', float32(4.4e+05)),
            ('pythonFloatScalar',  1.1),
            ('pythonFloatBigScalar', 5.5e+05)
        ]
        for datum in DATA:
            k, v = datum
            rdd = self.sc.parallelize([(0, v)])
            data = Data(rdd, nrecords=1, dtype='float32')
            setattr(self, k, v)
            setattr(self, k+"RDD", rdd)
            setattr(self, k+"Data", data)
        self.allCases = [datum[0] for datum in DATA]

    def test_casting(self):
        """Tests casting of numpy arrays, numpy scalars, and python scalars with Data.astype
        """
        isDowncastable = lambda name: 'Big' not in name

        for caseName in self.allCases:
            origVal, data = getattr(self, caseName), getattr(self, caseName+"Data")
            upcasted = data.astype('float64').first()[1]
            downcasted = data.astype('float16', casting="unsafe").first()[1]
            # upcasting to float64 shouldn't be a problem
            assert_true(allclose(origVal, upcasted, rtol=1e-05))
            if isDowncastable(caseName):
                # if the value is in range, we expect to get back something reasonably close to our original
                # values after downcasting to float16
                assert_true(allclose(origVal, downcasted, rtol=1e-03))
            else:
                # the unsafe cast will return *something*, but we don't have any expectations as to what:
                assert_is_not_none(downcasted)
            # raises py4j.protocol.Py4JJavaError after a TypeError on workers:
            # we're not importing py4j, and it seems like overkill to do so just for this one assertion,
            # so just assert an Exception.
            assert_raises(Exception, data.astype('float16', casting="safe").first)


class TestDataMethods(PySparkTestCase):

    def test_sortbykey(self):

        dataLocal = [
            ((0, 0), array([0])),
            ((0, 1), array([0])),
            ((0, 2), array([0])),
            ((1, 0), array([0])),
            ((1, 1), array([0])),
            ((1, 2), array([0]))
        ]

        data = Data(self.sc.parallelize(dataLocal))
        out = data.sortByKey().keys().collect()
        assert(array_equal(out, [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]))

        dataLocal = [
            ((0,), array([0])),
            ((1,), array([0])),
            ((2,), array([0]))
        ]

        data = Data(self.sc.parallelize(dataLocal))
        out = data.sortByKey().keys().collect()
        assert(array_equal(out, [(0,), (1,), (2,)]))

    def test_collect(self):

        dataLocal = [
            ((0, 0), array([0])),
            ((0, 1), array([1])),
            ((0, 2), array([2])),
            ((1, 0), array([3])),
            ((1, 1), array([4])),
            ((1, 2), array([5]))
        ]

        data = Data(self.sc.parallelize(dataLocal))

        out = data.collectKeysAsArray()

        assert(array_equal(out, [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]))

        out = data.collectValuesAsArray()

        assert(array_equal(out, [[0], [1], [2], [3], [4], [5]]))

    def test_collect_with_sorting(self):

        dataLocal = [
            ((0, 0), array([0])),
            ((0, 1), array([1])),
            ((0, 2), array([2])),
            ((1, 0), array([3])),
            ((1, 1), array([4])),
            ((1, 2), array([5]))
        ]

        data = Data(self.sc.parallelize(dataLocal))

        out = data.collectKeysAsArray(sorting=True)

        assert(array_equal(out, [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]))

        out = data.collectValuesAsArray(sorting=True)

        print(out)

        assert(array_equal(out, [[0], [3], [1], [4], [2], [5]]))

    def test_range_int_key(self):

        dataLocal = [
            (0, array([0])),
            (1, array([1])),
            (2, array([2])),
            (3, array([3])),
            (4, array([4])),
            (5, array([5]))
        ]

        data = Data(self.sc.parallelize(dataLocal))

        out = data.range(0, 2).collectKeysAsArray()
        assert(array_equal(out, [0, 1]))

        out = data.range(0, 5).collectKeysAsArray()
        assert(array_equal(out, [0, 1, 2, 3, 4]))

        out = data.range(0, 6).collectKeysAsArray()
        assert(array_equal(out, [0, 1, 2, 3, 4, 5]))

    def test_range_tuple_key(self):

        dataLocal = [
            ((0, 0), array([0])),
            ((0, 1), array([1])),
            ((0, 2), array([2])),
            ((1, 0), array([3])),
            ((1, 1), array([4])),
            ((1, 2), array([5]))
        ]

        data = Data(self.sc.parallelize(dataLocal))

        out = data.range((0, 0), (1, 1)).collectKeysAsArray()
        assert(array_equal(out, [(0, 0), (0, 1), (0, 2), (1, 0)]))
