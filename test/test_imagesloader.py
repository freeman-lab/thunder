import json
from numpy import arange, array_equal, ndarray
from numpy import dtype as dtypeFunc
import os
import unittest2 as unittest
from nose.tools import assert_almost_equal, assert_equals, assert_raises, assert_true

from thunder.rdds.fileio.imagesloader import ImagesLoader
from test_utils import PySparkTestCase, PySparkTestCaseWithOutputDir

_haveImage = False
try:
    from PIL import Image
    _haveImage = True
except ImportError:
    # PIL not available; skip tests that require it
    Image = None


class TestImagesFileLoaders(PySparkTestCase):
    @staticmethod
    def _findTestResourcesDir(resourcesdirname="resources"):
        testDirPath = os.path.dirname(os.path.realpath(__file__))
        testResourcesDirPath = os.path.join(testDirPath, resourcesdirname)
        if not os.path.isdir(testResourcesDirPath):
            raise IOError("Test resources directory "+testResourcesDirPath+" not found")
        return testResourcesDirPath

    def setUp(self):
        super(TestImagesFileLoaders, self).setUp()
        self.testResourcesDir = self._findTestResourcesDir()

    def test_fromArrays(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))

        image = ImagesLoader(self.sc).fromArrays(ary)

        collectedImage = image.collect()
        assert_equals(1, len(collectedImage))
        assert_equals(ary.shape, image.dims.count)
        assert_equals(0, collectedImage[0][0])  # check key
        assert_true(array_equal(ary, collectedImage[0][1]))  # check value

    def test_fromOCP(self):
        from urllib2 import urlopen, Request, URLError
        try:
            request = Request("http://ocp.me/ocp/ca/freeman14/info/")
            response = urlopen(request)
            imagePath = "freeman14"
            ocpImage = ImagesLoader(self.sc).fromOCP(imagePath, startIdx=0, stopIdx=1, minBound=(0, 0, 0),
                                                     maxBound=(128, 128, 16), resolution=0)
            assert_equals(ocpImage[1].shape, (128, 128, 16))
        except URLError, e:
            print "fromOCP is unavaliable"

    def test_fromPng(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_png", "dot1_grey.png")
        pngImage = ImagesLoader(self.sc).fromPng(imagePath)
        firstPngImage = pngImage.first()
        assert_equals(0, firstPngImage[0], "Key error; expected first image key to be 0, was "+str(firstPngImage[0]))
        expectedShape = (70, 75)
        assert_true(isinstance(firstPngImage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firstPngImage[1])))
        assert_equals(expectedShape, firstPngImage[1].shape)
        assert_equals(expectedShape, pngImage.dims.count)
        assert_almost_equal(0.937, firstPngImage[1].ravel().max(), places=2)  # integer val 239
        assert_almost_equal(0.00392, firstPngImage[1].ravel().min(), places=2)  # integer val 1

    def test_fromTif(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_tif", "dot1_grey_lzw.tif")
        tiffImage = ImagesLoader(self.sc).fromTif(imagePath)
        firstTiffImage = tiffImage.first()
        assert_equals(0, firstTiffImage[0], "Key error; expected first image key to be 0, was "+str(firstTiffImage[0]))
        expectedShape = (70, 75)
        assert_true(isinstance(firstTiffImage[1], ndarray),
                    "Value type error; expected first image value to be numpy ndarray, was " +
                    str(type(firstTiffImage[1])))
        assert_equals(expectedShape, firstTiffImage[1].shape)
        assert_equals(expectedShape, tiffImage.dims.count)
        assert_equals(239, firstTiffImage[1].ravel().max())
        assert_equals(1, firstTiffImage[1].ravel().min())

    @staticmethod
    def _evaluateMultipleImages(tiffImages, expectedNum, expectedShape, expectedKeys, expectedSums):
        assert_equals(expectedNum, len(tiffImages), "Expected %d images, got %d" % (expectedNum, len(tiffImages)))
        for img, expectedKey, expectedSum in zip(tiffImages, expectedKeys, expectedSums):
            assert_equals(expectedKey, img[0], "Expected key %s, got %s" % (str(expectedKey), str(img[0])))

            assert_true(isinstance(img[1], ndarray),
                        "Value type error; expected image value to be numpy ndarray, was " + str(type(img[1])))
            assert_equals(expectedShape, img[1].shape)
            assert_equals(expectedSum, img[1].sum())

    def test_fromTifWithMultipleFiles(self):
        imagePath = os.path.join(self.testResourcesDir, "singlelayer_tif", "dot*_grey_lzw.tif")
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath).collect()

        expectedNum = 3
        expectedShape = (70, 75)
        expectedSums = [1233881, 1212169, 1191300]
        # 3 images have increasing #s of black dots, so lower luminance overall
        expectedKeys = range(expectedNum)
        self._evaluateMultipleImages(tiffImages, expectedNum, expectedShape, expectedKeys, expectedSums)

    def _run_tst_multitif(self, filename, expectedDtype):
        imagePath = os.path.join(self.testResourcesDir, "multilayer_tif", filename)
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath).collect()

        expectedNum = 1
        expectedShape = (70, 75, 3)  # 3 concatenated pages, each with single luminance channel
        # 3 images have increasing #s of black dots, so lower luminance overall
        expectedSums = [1140006, 1119161, 1098917]
        expectedKey = 0

        assert_equals(expectedNum, len(tiffImages), "Expected %d images, got %d" % (expectedNum, len(tiffImages)))
        tiffImage = tiffImages[0]
        assert_equals(expectedKey, tiffImage[0], "Expected key %s, got %s" % (str(expectedKey), str(tiffImage[0])))
        assert_true(isinstance(tiffImage[1], ndarray),
                    "Value type error; expected image value to be numpy ndarray, was " + str(type(tiffImage[1])))
        assert_equals(expectedDtype, str(tiffImage[1].dtype))
        assert_equals(expectedShape, tiffImage[1].shape)
        for channelidx in xrange(0, expectedShape[2]):
            assert_equals(expectedSums[channelidx], tiffImage[1][:, :, channelidx].flatten().sum())

    @unittest.skipIf(not _haveImage, "PIL/pillow not installed or not functional")
    def test_fromMultipageTif(self):
        self._run_tst_multitif("dotdotdot_lzw.tif", "uint8")

    @unittest.skipIf(not _haveImage, "PIL/pillow not installed or not functional")
    def test_fromFloatingpointTif(self):
        self._run_tst_multitif("dotdotdot_float32.tif", "float32")

    def test_fromMultiTimepointTif(self):
        imagePath = os.path.join(self.testResourcesDir, "multilayer_tif", "dotdotdot_lzw.tif")
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath, nplanes=1)
        # we don't expect to have nrecords cached, since the driver doesn't know how many images there are per file
        assert_true(tiffImages._nrecords is None)
        assert_equals(3, tiffImages.nrecords)

        collectedTiffImages = tiffImages.collect()

        assert_equals(3, len(collectedTiffImages), "Expected 3 images, got %d" % len(collectedTiffImages))
        expectedSums = [1140006, 1119161, 1098917]
        expectedIdx = 0
        for idx, tiffAry in collectedTiffImages:
            assert_equals((70, 75), tiffAry.shape)
            assert_equals(expectedIdx, idx)
            assert_equals(expectedSums[idx], tiffAry.ravel().sum())
            expectedIdx += 1

    def test_fromMultipleMultiTimepointTifs(self):
        imagePath = os.path.join(self.testResourcesDir, "multilayer_tif", "dotdotdot_lzw*.tif")

        tiffImages = ImagesLoader(self.sc).fromTif(imagePath, nplanes=1)
        assert_true(tiffImages._nrecords is None)
        assert_equals(6, tiffImages.nrecords)

        collectedTiffImages = tiffImages.collect()

        assert_equals(6, len(collectedTiffImages), "Expected 6 images, got %d" % len(collectedTiffImages))
        expectedSums = [1140006, 1119161, 1098917, 1140006, 1119161, 1098917]
        expectedIdx = 0
        for idx, tiffAry in collectedTiffImages:
            assert_equals((70, 75), tiffAry.shape)
            assert_equals(expectedIdx, idx)
            assert_equals(expectedSums[idx], tiffAry.ravel().sum())
            expectedIdx += 1

        # 3 pages / file is not evenly divisible by 2 planes
        # note this will still log a big traceback, since the exception is in the executor,
        # not the driver. But this is expected behavior.
        assert_raises(Exception, ImagesLoader(self.sc).fromTif(imagePath, nplanes=2).count)

    def test_fromSignedIntTif(self):
        imagePath = os.path.join(self.testResourcesDir, "multilayer_tif", "test_signed.tif")
        tiffImages = ImagesLoader(self.sc).fromTif(imagePath, nplanes=1)
        assert_equals(2, tiffImages.nrecords)
        assert_equals('int16', tiffImages.dtype)

        collectedTiffImages = tiffImages.collect()

        assert_equals(2, len(collectedTiffImages), "Expected 2 images, got %d" % len(collectedTiffImages))
        expectedSums = [1973201, 2254767]
        expectedIdx = 0
        for idx, tiffAry in collectedTiffImages:
            assert_equals((120, 120), tiffAry.shape)
            assert_equals('int16', str(tiffAry.dtype))
            assert_equals(expectedIdx, idx)
            assert_equals(expectedSums[idx], tiffAry.ravel().sum())
            expectedIdx += 1


class TestImagesLoaderUsingOutputDir(PySparkTestCaseWithOutputDir):
    def test_fromStack(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test.stack")
        ary.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(filename, dims=(4, 2))

        collectedImage = image.collect()
        assert_equals(1, len(collectedImage))
        assert_equals(0, collectedImage[0][0])  # check key
        # assert that image shape *matches* that in image dimensions:
        assert_equals(image.dims.count, collectedImage[0][1].shape)
        assert_true(array_equal(ary.T, collectedImage[0][1]))  # check value

    def test_fromStacks(self):
        ary = arange(8, dtype=dtypeFunc('int16')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('int16')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test01.stack")
        ary.tofile(filename)
        filename = os.path.join(self.outputdir, "test02.stack")
        ary2.tofile(filename)

        image = ImagesLoader(self.sc).fromStack(self.outputdir, dims=(4, 2))

        collectedImage = image.collect()
        assert_equals(2, len(collectedImage))
        assert_equals(0, collectedImage[0][0])  # check key
        assert_equals(image.dims.count, collectedImage[0][1].shape)
        assert_true(array_equal(ary.T, collectedImage[0][1]))  # check value
        assert_equals(1, collectedImage[1][0])  # check image 2
        assert_true(array_equal(ary2.T, collectedImage[1][1]))

    def test_fromStacksWithConf(self):
        ary = arange(8, dtype=dtypeFunc('int32')).reshape((2, 4))
        ary2 = arange(8, 16, dtype=dtypeFunc('int32')).reshape((2, 4))
        filename = os.path.join(self.outputdir, "test01.stack")
        ary.tofile(filename)
        filename = os.path.join(self.outputdir, "test02.stack")
        ary2.tofile(filename)
        conf = {"dims": [4, 2], "dtype": "int32"}
        with open(os.path.join(self.outputdir, "conf.json"), 'w') as fp:
            json.dump(conf, fp)

        image = ImagesLoader(self.sc).fromStack(self.outputdir)
        assert_equals("int32", image._dtype)
        assert_equals(2, image._nrecords)
        assert_equals((4, 2), image._dims.count)

        collectedImage = image.collect()
        assert_equals(2, len(collectedImage))
        assert_equals(0, collectedImage[0][0])  # check key
        assert_equals(image.dims.count, collectedImage[0][1].shape)
        assert_true(array_equal(ary.T, collectedImage[0][1]))  # check value
        assert_equals(1, collectedImage[1][0])  # check image 2
        assert_true(array_equal(ary2.T, collectedImage[1][1]))

    def test_fromMultiTimepointStacks(self):
        ary = arange(16, dtype=dtypeFunc('uint8')).reshape((4, 2, 2))
        ary2 = arange(16, 32, dtype=dtypeFunc('uint8')).reshape((4, 2, 2))
        ary.tofile(os.path.join(self.outputdir, "test01.stack"))
        ary2.tofile(os.path.join(self.outputdir, "test02.stack"))

        image = ImagesLoader(self.sc).fromStack(self.outputdir, dtype="uint8", dims=(2, 2, 4), nplanes=2)
        collectedImage = image.collect()

        # we don't expect to have nrecords cached, since we get an unknown number of images per file
        assert_true(image._nrecords is None)
        assert_equals(4, image.nrecords)
        assert_equals(4, len(collectedImage))
        # check keys:
        assert_equals(0, collectedImage[0][0])
        assert_equals(1, collectedImage[1][0])
        assert_equals(2, collectedImage[2][0])
        assert_equals(3, collectedImage[3][0])
        # check values:
        assert_true(array_equal(ary[:2].T, collectedImage[0][1]))
        assert_true(array_equal(ary[2:].T, collectedImage[1][1]))
        assert_true(array_equal(ary2[:2].T, collectedImage[2][1]))
        assert_true(array_equal(ary2[2:].T, collectedImage[3][1]))

        # 3 planes does not divide 4
        assert_raises(ValueError, ImagesLoader(self.sc).fromStack, self.outputdir, dtype="uint8",
                      dims=(2, 2, 4), nplanes=3)
