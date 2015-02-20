import glob
import json
import os
import struct
import unittest

from nose.tools import assert_almost_equal, assert_equals, assert_true, assert_raises
from numpy import allclose, arange, array, array_equal
from numpy import dtype as dtypeFunc

from thunder.rdds.fileio.seriesloader import SeriesLoader
from thunder.utils.common import smallestFloatType
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class SeriesBinaryTestData(object):
    """
    Data object for SeriesLoader binary test.
    """
    __slots__ = ('keys', 'vals', 'keyDtype', 'valDtype')

    def __init__(self, keys, vals, keyDtype, valDtype):
        """
        Constructor, intended to be called from fromArrays class factory method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
        self.keys = keys
        self.vals = vals
        self.keyDtype = keyDtype
        self.valDtype = valDtype

    @property
    def keyStructFormat(self):
        return self.keyDtype.char * self.nkeys

    @property
    def valStructFormat(self):
        return self.valDtype.char * self.nvals

    @property
    def data(self):
        return zip(self.keys, self.vals)

    @property
    def nkeys(self):
        return len(self.keys[0])

    @property
    def nvals(self):
        return len(self.vals[0])

    def writeToFile(self, f):
        """
        Writes own key, value data to passed file handle in binary format
        Parameters
        ----------
        f: file handle, open for writing
            f will remain open after this call
        """
        for keys, vals in self.data:
            f.write(struct.pack(self.keyStructFormat, *keys))
            f.write(struct.pack(self.valStructFormat, *vals))

    @staticmethod
    def _validateLengths(dat):
        l = len(dat[0])
        for d in dat:
            assert len(d) == l, "Data of unequal lengths, %d and %d" % (l, len(d))

    @staticmethod
    def _normalizeDType(dtypeInstance, data):
        if dtypeInstance is None:
            return data.dtype
        return dtypeFunc(dtypeInstance)

    @classmethod
    def fromArrays(cls, keys, vals, keyDtype=None, valDtype=None):
        """
        Factory method for SeriesBinaryTestData. Validates input before calling class __init__ method.

        Expects m x n and m x p data for keys and vals.

        Parameters
        ----------
        keys: two dimensional array or sequence
        vals: two dimensional array or sequence
        keydtype: object castable to numpy dtype
            data type of keys
        valdtype: object castable to numpy dtype
            data type of values

        Returns
        -------
        self: new instance of SeriesBinaryTestData
        """
        keyDtype = cls._normalizeDType(keyDtype, keys)
        valDtype = cls._normalizeDType(valDtype, vals)
        assert len(keys) == len(vals), "Unequal numbers of keys and values, %d and %d" % (len(keys), len(vals))
        cls._validateLengths(keys)
        cls._validateLengths(vals)
        return cls(keys, vals, keyDtype, valDtype)


class TestSeriesLoader(PySparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesDirName="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesDirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    @staticmethod
    def _findSourceTreeDir(dirName="utils/data"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, "..", "thunder", dirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def test_fromArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays(ary)

        seriesVals = series.collect()
        seriesAry = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((2, 0), seriesVals[2][0])
        assert_equals((3, 0), seriesVals[3][0])
        assert_equals((0, 1), seriesVals[4][0])
        assert_equals((1, 1), seriesVals[5][0])
        assert_equals((2, 1), seriesVals[6][0])
        assert_equals((3, 1), seriesVals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16')).ravel()
        assert_true(array_equal(ary.ravel(), collectedVals))

        # check that packing returns transpose of original array
        assert_true(array_equal(ary.T, seriesAry))

    def test_fromMultipleArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('int16')).reshape((2, 4))

        series = SeriesLoader(self.sc).fromArrays([ary, ary2])

        seriesVals = series.collect()
        seriesAry = series.pack()

        # check ordering of keys
        assert_equals((0, 0), seriesVals[0][0])  # first key
        assert_equals((1, 0), seriesVals[1][0])  # second key
        assert_equals((3, 0), seriesVals[3][0])
        assert_equals((0, 1), seriesVals[4][0])
        assert_equals((3, 1), seriesVals[7][0])

        # check dimensions tuple is reversed from numpy shape
        assert_equals(ary.shape[::-1], series.dims.count)

        # check that values are in original order, with subsequent point concatenated in values
        collectedVals = array([kv[1] for kv in seriesVals], dtype=dtypeFunc('int16'))
        assert_true(array_equal(ary.ravel(), collectedVals[:, 0]))
        assert_true(array_equal(ary2.ravel(), collectedVals[:, 1]))

        # check that packing returns concatenation of input arrays, with time as first dimension
        assert_true(array_equal(ary.T, seriesAry[0]))
        assert_true(array_equal(ary2.T, seriesAry[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromMultipageTif(self):
        testResourcesDir = TestSeriesLoader._findTestResourcesDir()
        imagePath = os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif")

        testImg_Pil = Image.open(imagePath)
        testImgArys = list()
        testImgArys.append(array(testImg_Pil))
        testImg_Pil.seek(1)
        testImgArys.append(array(testImg_Pil))
        testImg_Pil.seek(2)
        testImgArys.append(array(testImg_Pil))

        series = SeriesLoader(self.sc).fromTif(imagePath)
        assert_equals('float16', series._dtype)
        seriesAry = series.pack()

        assert_equals((70, 75, 3), series.dims.count)
        assert_equals((70, 75, 3), seriesAry.shape)
        assert_true(array_equal(testImgArys[0], seriesAry[:, :, 0]))
        assert_true(array_equal(testImgArys[1], seriesAry[:, :, 1]))
        assert_true(array_equal(testImgArys[2], seriesAry[:, :, 2]))

    def _run_fromFishTif(self, blocksize):
        imagepath = TestSeriesLoader._findSourceTreeDir("utils/data/fish/tif-stack")
        series = SeriesLoader(self.sc).fromTif(imagepath, blockSize=blocksize)
        assert_equals('float16', series._dtype)
        seriesAry = series.pack()
        seriesAry_xpose = series.pack(transpose=True)
        assert_equals('float16', str(seriesAry.dtype))
        assert_equals((76, 87, 2), series.dims.count)
        assert_equals((20, 76, 87, 2), seriesAry.shape)
        assert_equals((20, 2, 87, 76), seriesAry_xpose.shape)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromFishTif(self):
        self._run_fromFishTif("150M")

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_fromFishTifWithTinyBlocks(self):
        self._run_fromFishTif(blocksize=76*20)


class TestSeriesLoaderFromStacks(PySparkTestCaseWithOutputDir):
    def test_loadStacksAsSeries(self):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        rangeAry.shape = (64, 128)
        filepath = os.path.join(self.outputdir, "rangeAry.stack")
        rangeAry.tofile(filepath)

        series = SeriesLoader(self.sc).fromStack(filepath, dims=(128, 64))
        seriesAry = series.pack()

        assert_equals((128, 64), series.dims.count)
        assert_equals((128, 64), seriesAry.shape)
        assert_true(array_equal(rangeAry.T, seriesAry))


class TestSeriesBinaryLoader(PySparkTestCaseWithOutputDir):

    def _run_tst_fromBinary(self, useConfJson=False):
        # run this as a single big test so as to avoid repeated setUp and tearDown of the spark context
        # data will be a sequence of test data
        # all keys and all values in a test data item must be of the same length
        # keys get converted to ints regardless of raw input format
        DATA = [
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3], [5, 6, 7]], [[11], [12]], 'int16', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int16', 'int32'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11, 12, 13]], 'int32', 'int16'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'int16', 'float32'),
            SeriesBinaryTestData.fromArrays([[1, 2, 3]], [[11.0, 12.0, 13.0]], 'float32', 'float32'),
            SeriesBinaryTestData.fromArrays([[2, 3, 4]], [[11.0, 12.0, 13.0]], 'float32', 'float32'),
        ]

        for itemidx, item in enumerate(DATA):
            outSubdir = os.path.join(self.outputdir, 'input%d' % itemidx)
            os.mkdir(outSubdir)

            fname = os.path.join(outSubdir, 'inputfile%d.bin' % itemidx)
            with open(fname, 'wb') as f:
                item.writeToFile(f)

            loader = SeriesLoader(self.sc)
            if not useConfJson:
                series = loader.fromBinary(outSubdir, nkeys=item.nkeys, nvalues=item.nvals, keyType=str(item.keyDtype),
                                           valueType=str(item.valDtype))
            else:
                # write configuration file
                conf = {'input': outSubdir,
                        'nkeys': item.nkeys, 'nvalues': item.nvals,
                        'valuetype': str(item.valDtype), 'keytype': str(item.keyDtype)}
                with open(os.path.join(outSubdir, "conf.json"), 'wb') as f:
                    json.dump(conf, f, indent=2)
                series = loader.fromBinary(outSubdir)

            seriesData = series.rdd.collect()

            expectedData = item.data
            assert_equals(len(expectedData), len(seriesData),
                          "Differing numbers of k/v pairs in item %d; expected %d, got %d" %
                          (itemidx, len(expectedData), len(seriesData)))

            for expected, actual in zip(expectedData, seriesData):
                expectedKeys = tuple(expected[0])
                expectedType = smallestFloatType(item.valDtype)
                expectedVals = array(expected[1], dtype=expectedType)
                assert_equals(expectedKeys, actual[0],
                              "Key mismatch in item %d; expected %s, got %s" %
                              (itemidx, str(expectedKeys), str(actual[0])))
                assert_true(allclose(expectedVals, actual[1]),
                            "Value mismatch in item %d; expected %s, got %s" %
                            (itemidx, str(expectedVals), str(actual[1])))
                assert_equals(expectedType, str(actual[1].dtype),
                              "Value type mismatch in item %d; expected %s, got %s" %
                              (itemidx, expectedType, str(actual[1].dtype)))

    def test_fromBinary(self):
        self._run_tst_fromBinary()

    def test_fromBinaryWithConfFile(self):
        self._run_tst_fromBinary(True)


class TestSeriesBinaryWriteFromStack(PySparkTestCaseWithOutputDir):

    def _run_roundtrip_tst(self, testCount, arrays, blockSize):
        #  print "Running TestSeriesBinaryWriteFromStack roundtrip test #%d" % testCount
        inSubdir = os.path.join(self.outputdir, 'input%d' % testCount)
        os.mkdir(inSubdir)

        outSubdir = os.path.join(self.outputdir, 'output%d' % testCount)
        # os.mkdir(outSubdir)

        for aryCount, ary in enumerate(arrays):
            # array.tofile always writes in column-major order...
            ary.tofile(os.path.join(inSubdir, "img%02d.stack" % aryCount))

        # ... but we will read and interpret these as though they are in row-major order
        dims = list(arrays[0].shape)
        dims.reverse()

        underTest = SeriesLoader(self.sc)

        underTest.saveFromStack(inSubdir, outSubdir, dims, blockSize=blockSize, dtype=str(arrays[0].dtype))
        series = underTest.fromStack(inSubdir, dims, dtype=str(arrays[0].dtype))

        roundtrippedSeries = underTest.fromBinary(outSubdir)
        roundtripped = roundtrippedSeries.collect()
        direct = series.collect()

        expectedDtype = str(smallestFloatType(arrays[0].dtype))
        assert_equals(expectedDtype, roundtrippedSeries.dtype)
        assert_equals(expectedDtype, series.dtype)
        assert_equals(expectedDtype, str(roundtripped[0][1].dtype))
        assert_equals(expectedDtype, str(direct[0][1].dtype))

        with open(os.path.join(outSubdir, "conf.json"), 'r') as fp:
            # check that binary series file data type *matches* input stack data type (not yet converted to float)
            # at least according to conf.json
            conf = json.load(fp)
            assert_equals(str(arrays[0].dtype), conf["valuetype"])

        for ((seriesKeys, seriesValues), (directKeys, directValues)) in zip(roundtripped, direct):
            assert_equals(directKeys, seriesKeys)
            assert_equals(directValues, seriesValues)

            for seriesIdx, seriesVal in enumerate(seriesValues):
                # print "seriesIdx: %d; seriesKeys: %s; seriesVal: %g" % (seriesIdx, seriesKeys, seriesVal)
                # flip indices again for row vs col-major insanity
                aryKeys = list(seriesKeys)
                aryKeys.reverse()
                msg = "Failure on test #%d, time point %d, indices %s" % (testCount, seriesIdx, str(tuple(aryKeys)))
                try:
                    assert_almost_equal(arrays[seriesIdx][tuple(aryKeys)], seriesVal, places=4)
                except AssertionError, e:
                    raise AssertionError(msg, e)

    @staticmethod
    def generateTestImages(narrays, dims, datatype):
        nimgvals = reduce(lambda x, y: x * y, dims)
        return [arange(i*nimgvals, (i+1)*nimgvals, dtype=datatype).reshape(dims) for i in xrange(narrays)]

    def _roundtrip_tst_driver(self, moreTests=False):
        # parameterized test fixture
        arrays = TestSeriesBinaryWriteFromStack.generateTestImages(1, (2, 3), "int16")
        self._run_roundtrip_tst(0, arrays, 6*2)
        self._run_roundtrip_tst(1, arrays, 2*2)
        self._run_roundtrip_tst(2, arrays, 5*2)
        self._run_roundtrip_tst(3, arrays, 7*2)

        if moreTests:
            arrays = TestSeriesBinaryWriteFromStack.generateTestImages(3, (5, 7, 5), "int16")
            self._run_roundtrip_tst(4, arrays, 3*5*2)

            arrays = TestSeriesBinaryWriteFromStack.generateTestImages(3, (5, 7, 5), "int32")
            self._run_roundtrip_tst(5, arrays, 3*5*4)
            self._run_roundtrip_tst(6, arrays, 5*7*4)
            self._run_roundtrip_tst(7, arrays, 3*4)

            arrays = TestSeriesBinaryWriteFromStack.generateTestImages(3, (2, 4, 6), "float32")
            self._run_roundtrip_tst(8, arrays, 5*4)

    def test_roundtrip(self):
        self._roundtrip_tst_driver(False)


class TestSeriesBinaryRoundtrip(PySparkTestCaseWithOutputDir):

    def _run_roundtrip_tst(self, testIdx, nimages, aryShape, dtypeSpec, npartitions):
        testArrays = TestSeriesBinaryWriteFromStack.generateTestImages(nimages, aryShape, dtypeSpec)
        loader = SeriesLoader(self.sc)
        series = loader.fromArrays(testArrays)

        saveDirPath = os.path.join(self.outputdir, 'save%d' % testIdx)
        series.repartition(npartitions)  # note: this does an elementwise shuffle! won't be in sorted order
        series.saveAsBinarySeries(saveDirPath)

        nnonemptyPartitions = 0
        for partitionList in series.rdd.glom().collect():
            if partitionList:
                nnonemptyPartitions += 1
        del partitionList
        nsaveFiles = len(glob.glob(saveDirPath + os.sep + "*.bin"))

        roundtrippedSeries = loader.fromBinary(saveDirPath)

        with open(os.path.join(saveDirPath, "conf.json"), 'r') as fp:
            conf = json.load(fp)

        # sorting is required here b/c of the randomization induced by the repartition.
        # orig and roundtripped will in general be different from each other, since roundtripped
        # will have (0, 0, 0) index as first element (since it will be the lexicographically first
        # file) while orig has only a 1 in npartitions chance of starting with (0, 0, 0) after repartition.
        expectedPackedAry = series.pack(sorting=True)
        actualPackedAry = roundtrippedSeries.pack(sorting=True)

        assert_true(array_equal(expectedPackedAry, actualPackedAry))

        assert_equals(nnonemptyPartitions, nsaveFiles)

        assert_equals(len(aryShape), conf["nkeys"])
        assert_equals(nimages, conf["nvalues"])
        assert_equals("int16", conf["keytype"])
        assert_equals(str(series.dtype), conf["valuetype"])
        # check that we have converted ourselves to an appropriate float after reloading
        assert_equals(str(smallestFloatType(series.dtype)), str(roundtrippedSeries.dtype))

    def test_roundtrip(self):
        self._run_roundtrip_tst(0, 2, (2, 5, 5), "int16", 1)
        self._run_roundtrip_tst(1, 2, (2, 5, 5), "float32", 2)
        self._run_roundtrip_tst(2, 4, (5, 25, 25), "float16", 5)


class TestSeriesBlocksRoundtrip(PySparkTestCase):

    def _run_roundtrip_tst(self, nimages, aryShape, dtypeSpec, sizeSpec):
        testArrays = TestSeriesBinaryWriteFromStack.generateTestImages(nimages, aryShape, dtypeSpec)
        loader = SeriesLoader(self.sc)
        series = loader.fromArrays(testArrays)

        blocks = series.toBlocks(sizeSpec)

        roundtrippedSeries = blocks.toSeries(newDType=series.dtype)

        packedSeries = series.pack()
        packedRoundtrippedSeries = roundtrippedSeries.pack()

        assert_true(array_equal(packedSeries, packedRoundtrippedSeries))

    def _run_roundtrip_exception_tst(self, nimages, aryShape, dtypeSpec, sizeSpec):
        testArrays = TestSeriesBinaryWriteFromStack.generateTestImages(nimages, aryShape, dtypeSpec)
        loader = SeriesLoader(self.sc)
        series = loader.fromArrays(testArrays)

        assert_raises(ValueError, series.toBlocks, sizeSpec)

    def test_roundtrip(self):
        self._run_roundtrip_tst(2, (2, 5, 5), "int16", (1, 1, 2))
        self._run_roundtrip_tst(2, (2, 5, 5), "int16", (1, 5, 2))
        self._run_roundtrip_tst(2, (2, 5, 5), "int16", (5, 5, 2))
        self._run_roundtrip_tst(2, (2, 5, 5), "uint8", 20)

        self._run_roundtrip_exception_tst(2, (2, 5, 5), "int16", (1, 5, 1))
