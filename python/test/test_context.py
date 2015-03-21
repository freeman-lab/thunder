import json
import os
import unittest
import json
import tempfile
from nose.tools import assert_equals, assert_true
from numpy import arange, array, array_equal, mod
from numpy import dtype as dtypeFunc

from test_utils import PySparkTestCaseWithOutputDir
from thunder import ThunderContext

_have_image = False
try:
    from PIL import Image
    _have_image = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class TestContextLoading(PySparkTestCaseWithOutputDir):
    def setUp(self):
        super(TestContextLoading, self).setUp()
        self.tsc = ThunderContext(self.sc)

    @staticmethod
    def _findTestResourcesDir(resourcesDirName="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesDirName)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def __run_loadStacksAsSeries(self, shuffle):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((128, 64), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(filePath, dims=(128, 64), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)  # check before any potential first() calls update this val
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((128, 64), rangeSeries.dims.count)
        assert_equals((128, 64), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry))

    def test_loadStacksAsSeriesNoShuffle(self):
        self.__run_loadStacksAsSeries(False)

    def test_loadStacksAsSeriesWithShuffle(self):
        self.__run_loadStacksAsSeries(True)

    def __run_load3dStackAsSeries(self, shuffle):
        rangeAry = arange(32*64*4, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((32, 64, 4), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(filePath, dims=(32, 64, 4), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((32, 64, 4), rangeSeries.dims.count)
        assert_equals((32, 64, 4), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry))

    def test_load3dStackAsSeriesNoShuffle(self):
        self.__run_load3dStackAsSeries(False)

    def test_load3dStackAsSeriesWithShuffle(self):
        self.__run_load3dStackAsSeries(True)

    def __run_loadMultipleStacksAsSeries(self, shuffle, writeConf=False):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((128, 64), order='F')
        rangeAry2 = arange(64*128, 2*64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeAry2.tofile(filePath)
        expectedAry2 = rangeAry2.reshape((128, 64), order='F')
        if writeConf:
            conf = {"dims": (128, 64)}
            with open(os.path.join(self.outputdir, "alternateConfName.json"), 'w') as fp:
                json.dump(conf, fp)
            rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, shuffle=shuffle,
                                                      confFilename="alternateConfName.json")
        else:
            rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, dims=(128, 64), shuffle=shuffle)
        assert_equals('float32', rangeSeries._dtype)

        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((128, 64), rangeSeries.dims.count)
        assert_equals((2, 128, 64), rangeSeriesAry.shape)
        assert_equals((2, 64, 128), rangeSeriesAry_xpose.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry, rangeSeriesAry[0]))
        assert_true(array_equal(expectedAry2, rangeSeriesAry[1]))
        assert_true(array_equal(expectedAry.T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(expectedAry2.T, rangeSeriesAry_xpose[1]))

    def test_loadMultipleStacksAsSeriesNoShuffle(self):
        self.__run_loadMultipleStacksAsSeries(False)

    def test_loadMultipleStacksAsSeriesWithShuffle(self):
        self.__run_loadMultipleStacksAsSeries(True)

    def test_loadMultipleStacksAsSeriesWithConfFile(self):
        self.__run_loadMultipleStacksAsSeries(True, writeConf=True)

    def test_loadMultipleMultipointStacksAsSeries(self):
        rangeAry = arange(64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary01.stack")
        rangeAry.tofile(filePath)
        expectedAry = rangeAry.reshape((32, 32, 8), order='F')
        rangeAry2 = arange(64*128, 2*64*128, dtype=dtypeFunc('int16'))
        filePath = os.path.join(self.outputdir, "rangeary02.stack")
        rangeAry2.tofile(filePath)
        expectedAry2 = rangeAry2.reshape((32, 32, 8), order='F')

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, dims=(32, 32, 8), nplanes=2, shuffle=True)
        assert_equals('float32', rangeSeries._dtype)

        rangeSeriesAry = rangeSeries.pack()

        assert_equals((32, 32, 2), rangeSeries.dims.count)
        assert_equals((8, 32, 32, 2), rangeSeriesAry.shape)
        assert_equals('float32', str(rangeSeriesAry.dtype))
        assert_true(array_equal(expectedAry[:, :, :2], rangeSeriesAry[0]))
        assert_true(array_equal(expectedAry[:, :, 2:4], rangeSeriesAry[1]))
        assert_true(array_equal(expectedAry[:, :, 4:6], rangeSeriesAry[2]))
        assert_true(array_equal(expectedAry[:, :, 6:], rangeSeriesAry[3]))
        assert_true(array_equal(expectedAry2[:, :, :2], rangeSeriesAry[4]))
        assert_true(array_equal(expectedAry2[:, :, 2:4], rangeSeriesAry[5]))
        assert_true(array_equal(expectedAry2[:, :, 4:6], rangeSeriesAry[6]))
        assert_true(array_equal(expectedAry2[:, :, 6:], rangeSeriesAry[7]))

    def __run_loadTifAsSeries(self, shuffle):
        tmpAry = arange(60*120, dtype=dtypeFunc('uint16'))
        rangeAry = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry)
        filePath = os.path.join(self.outputdir, "rangetif01.tif")
        pilImg.save(filePath)
        del pilImg, tmpAry

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat="tif-stack", shuffle=shuffle)
        assert_equals('float16', rangeSeries._dtype)  # check before any potential first() calls update this val
        rangeSeriesAry = rangeSeries.pack()

        assert_equals((60, 120), rangeSeries.dims.count)  # 2d tif now loaded as 2d image; was 3d with singleton z dim
        assert_equals((60, 120), rangeSeriesAry.shape)
        assert_equals('float16', str(rangeSeriesAry.dtype))
        assert_true(array_equal(rangeAry, rangeSeriesAry))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesNoShuffle(self):
        self.__run_loadTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTifAsSeriesWithShuffle(self):
        self.__run_loadTifAsSeries(True)

    def __run_loadTestTifAsSeries(self, shuffle):
        testResourcesDir = TestContextLoading._findTestResourcesDir()
        imagePath = os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif")

        testimg_pil = Image.open(imagePath)
        testimg_arys = list()
        testimg_arys.append(array(testimg_pil))  # original shape 70, 75
        testimg_pil.seek(1)
        testimg_arys.append(array(testimg_pil))
        testimg_pil.seek(2)
        testimg_arys.append(array(testimg_pil))

        rangeSeries = self.tsc.loadImagesAsSeries(imagePath, inputFormat="tif-stack", shuffle=shuffle)
        assert_true(rangeSeries._dtype.startswith("float"))
        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((70, 75, 3), rangeSeries.dims.count)
        assert_equals((70, 75, 3), rangeSeriesAry.shape)
        assert_equals((3, 75, 70), rangeSeriesAry_xpose.shape)
        assert_true(rangeSeriesAry.dtype.kind == "f")
        assert_true(array_equal(testimg_arys[0], rangeSeriesAry[:, :, 0]))
        assert_true(array_equal(testimg_arys[1], rangeSeriesAry[:, :, 1]))
        assert_true(array_equal(testimg_arys[2], rangeSeriesAry[:, :, 2]))
        assert_true(array_equal(testimg_arys[0].T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(testimg_arys[1].T, rangeSeriesAry_xpose[1]))
        assert_true(array_equal(testimg_arys[2].T, rangeSeriesAry_xpose[2]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesNoShuffle(self):
        self.__run_loadTestTifAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadTestTifAsSeriesWithShuffle(self):
        self.__run_loadTestTifAsSeries(True)

    def __run_loadMultipleTifsAsSeries(self, shuffle):
        tmpAry = arange(60*120, dtype=dtypeFunc('uint16'))
        rangeAry = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry)
        filePath = os.path.join(self.outputdir, "rangetif01.tif")
        pilImg.save(filePath)

        tmpAry = arange(60*120, 2*60*120, dtype=dtypeFunc('uint16'))
        rangeAry2 = mod(tmpAry, 255).astype('uint8').reshape((60, 120))
        pilImg = Image.fromarray(rangeAry2)
        filePath = os.path.join(self.outputdir, "rangetif02.tif")
        pilImg.save(filePath)

        del pilImg, tmpAry

        rangeSeries = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat="tif-stack", shuffle=shuffle)
        assert_equals('float16', rangeSeries._dtype)
        rangeSeriesAry = rangeSeries.pack()
        rangeSeriesAry_xpose = rangeSeries.pack(transpose=True)

        assert_equals((60, 120), rangeSeries.dims.count)  # 2d tif now loaded as 2d image; was 3d with singleton z dim
        assert_equals((2, 60, 120), rangeSeriesAry.shape)
        assert_equals((2, 120, 60), rangeSeriesAry_xpose.shape)
        assert_equals('float16', str(rangeSeriesAry.dtype))
        assert_true(array_equal(rangeAry, rangeSeriesAry[0]))
        assert_true(array_equal(rangeAry2, rangeSeriesAry[1]))
        assert_true(array_equal(rangeAry.T, rangeSeriesAry_xpose[0]))
        assert_true(array_equal(rangeAry2.T, rangeSeriesAry_xpose[1]))

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesNoShuffle(self):
        self.__run_loadMultipleTifsAsSeries(False)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleTifsAsSeriesWithShuffle(self):
        self.__run_loadMultipleTifsAsSeries(True)

    @unittest.skipIf(not _have_image, "PIL/pillow not installed or not functional")
    def test_loadMultipleMultipointTifsAsSeries(self):
        testResourcesDir = TestContextLoading._findTestResourcesDir()
        imagesPath = os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw*.tif")

        # load only one file, second is a copy of this one
        testimg_pil = Image.open(os.path.join(testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif"))
        testimg_arys = [array(testimg_pil)]
        for idx in xrange(1, 3):
            testimg_pil.seek(idx)
            testimg_arys.append(array(testimg_pil))

        rangeSeries = self.tsc.loadImagesAsSeries(imagesPath, inputFormat="tif-stack", shuffle=True, nplanes=1)
        assert_equals((70, 75), rangeSeries.dims.count)

        rangeSeriesAry = rangeSeries.pack()
        assert_equals((6, 70, 75), rangeSeriesAry.shape)
        for idx in xrange(6):
            assert_true(array_equal(testimg_arys[idx % 3], rangeSeriesAry[idx]))

    @staticmethod
    def _tempFileWithPaths(f, blob):
        f.write(blob)
        f.flush()
        return f.name

    def test_loadParams(self):

        params = json.dumps({"name": "test1", "value": [1, 2, 3]})

        f = tempfile.NamedTemporaryFile()
        path = TestContextLoading._tempFileWithPaths(f, params)

        d = self.tsc.loadParams(path)

        assert(d.names() == ["test1"])
        assert(array_equal(d.values(), [1, 2, 3]))

        params = json.dumps([{"name": "test0", "value": [1, 2, 3]},
                             {"name": "test1", "value": [4, 5, 6]}])

        f = tempfile.NamedTemporaryFile()
        path = TestContextLoading._tempFileWithPaths(f, params)
        d = self.tsc.loadParams(path)

        assert(d.names() == ["test0", "test1"])
        assert(array_equal(d.values(), [[1, 2, 3], [4, 5, 6]]))
        assert(array_equal(d.values("test0"), [1, 2, 3]))


class TestContextWriting(PySparkTestCaseWithOutputDir):

    def setUp(self):
        super(TestContextWriting, self).setUp()
        self.tsc = ThunderContext(self.sc)

    def test_export_npy(self):

        from numpy import load

        a = array([[1, 2], [2, 3]])

        filename = self.outputdir + "/test.npy"
        self.tsc.export(a, filename)
        aa = load(filename)
        assert(array_equal(aa, a))

        filename = self.outputdir + "/test"
        self.tsc.export(a, filename, format="npy", overwrite=True)
        aa = load(filename + ".npy")
        assert(array_equal(aa, a))

    def test_export_mat(self):

        from scipy.io import loadmat

        a = array([[1, 2], [2, 3]])

        filename = self.outputdir + "/test.mat"
        self.tsc.export(a, filename)
        aa = loadmat(filename)
        assert(array_equal(aa['test'], a))

        filename = self.outputdir + "/test"
        self.tsc.export(a, filename, format="mat", overwrite=True)
        aa = loadmat(filename + ".mat")
        assert(array_equal(aa['test'], a))

        filename = self.outputdir + "/test"
        self.tsc.export(a, filename, format="mat", varname="tmp", overwrite=True)
        aa = loadmat(filename + ".mat")
        assert(array_equal(aa['tmp'], a))

    def test_export_txt(self):

        from numpy import loadtxt

        a = array([[1, 2], [2, 3]])

        filename = self.outputdir + "/test.txt"
        self.tsc.export(a, filename)
        aa = loadtxt(filename)
        assert(array_equal(aa, a))

        filename = self.outputdir + "/test"
        self.tsc.export(a, filename, format="txt", overwrite=True)
        aa = loadtxt(filename + ".txt")
        assert(array_equal(aa, a))


class TestLoadIrregularImages(PySparkTestCaseWithOutputDir):
    def setUp(self):
        super(TestLoadIrregularImages, self).setUp()
        self.tsc = ThunderContext(self.sc)

    def _generate_array(self, dtype):
        self.ary = arange(256, dtype=dtypeFunc(dtype)).reshape((16, 4, 4))  # 16 pages of 4x4 images

    def _write_tiffs(self):
        import thunder.rdds.fileio.tifffile as tifffile
        writer1 = tifffile.TiffWriter(os.path.join(self.outputdir, "tif01.TIF"))
        writer1.save(self.ary[:8].transpose((0, 2, 1)), photometric="minisblack")  # write out 8 pages
        writer1.close()
        del writer1

        writer2 = tifffile.TiffWriter(os.path.join(self.outputdir, "tif02.TIF"))
        writer2.save(self.ary.transpose((0, 2, 1)), photometric="minisblack")  # write out all 16 pages
        writer2.close()
        del writer2

    def _write_stacks(self):
        with open(os.path.join(self.outputdir, "stack01.bin"), "w") as f:
            self.ary[:8].tofile(f)
        with open(os.path.join(self.outputdir, "stack02.bin"), "w") as f:
            self.ary.tofile(f)

    def _run_tst(self, imgType, dtype, writeConf=False):
        self._generate_array(dtype)
        if imgType.lower().startswith('tif'):
            self._write_tiffs()
            inputFormat, ext, dims = "tif", "TIF", None
        elif imgType.lower().startswith("stack"):
            self._write_stacks()
            inputFormat, ext, dims = "stack", "bin", (16, 4, 4)
        else:
            raise ValueError("Unknown imgType: %s" % imgType)

        if writeConf:
            conf = {"ext": ext, "dims": dims, "nplanes": 2}
            with open(os.path.join(self.outputdir, "conf.json"), "w") as fp:
                json.dump(conf, fp)
            series = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat=inputFormat,
                                                 blockSize=(2, 1, 1), blockSizeUnits="pixels",
                                                 renumber=True)
        else:
            # with nplanes=2, this should yield a 12 record Images object, which after converting to
            # a series and packing should be a 12 x 4 x 4 x 2 array.
            # renumber=True is required in this case in order to ensure sensible results.
            series = self.tsc.loadImagesAsSeries(self.outputdir, inputFormat=inputFormat, ext=ext,
                                                 blockSize=(2, 1, 1), blockSizeUnits="pixels",
                                                 nplanes=2, dims=dims, renumber=True)
        packedAry = series.pack()
        assert_equals((12, 4, 4, 2), packedAry.shape)
        assert_true(array_equal(self.ary[0:2], packedAry[0].T))
        assert_true(array_equal(self.ary[2:4], packedAry[1].T))
        assert_true(array_equal(self.ary[4:6], packedAry[2].T))
        assert_true(array_equal(self.ary[6:8], packedAry[3].T))  # first image was only 4 2-plane records
        assert_true(array_equal(self.ary[0:2], packedAry[4].T))
        assert_true(array_equal(self.ary[2:4], packedAry[5].T))
        assert_true(array_equal(self.ary[4:6], packedAry[6].T))
        assert_true(array_equal(self.ary[6:8], packedAry[7].T))
        assert_true(array_equal(self.ary[8:10], packedAry[8].T))
        assert_true(array_equal(self.ary[10:12], packedAry[9].T))
        assert_true(array_equal(self.ary[12:14], packedAry[10].T))
        assert_true(array_equal(self.ary[14:16], packedAry[11].T))

    def test_loadMultipleSignedIntTifsAsSeries(self):
        self._run_tst('tif', 'int16')

    def test_loadMultipleUnsignedIntTifsAsSeries(self):
        self._run_tst('tif', 'uint16')

    def test_loadMultipleTifsAsSeriesWithConf(self):
        self._run_tst('tif', 'uint16', writeConf=True)

    # can't currently have binary stack files of different sizes, since we have
    # fixed `dims` for all stacks. leaving in place b/c it seems like something
    # to support soon.
    # def test_loadMultipleBinaryStacksAsSeries(self):
    #    self._run_tst('stack', 'uint16')