import shutil
import tempfile

from test_utils import PySparkTestCase
from numpy import allclose, dstack, mean, random, ndarray, zeros, pi
from scipy.ndimage.interpolation import shift
from nose.tools import assert_true

from thunder.registration.registration import Registration
from thunder.registration.transformation import TranslationTransformation, EuclideanTransformation
from thunder.rdds.fileio.imagesloader import ImagesLoader


class ImgprocessingTestCase(PySparkTestCase):
    def setUp(self):
        super(ImgprocessingTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ImgprocessingTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRegistrationBasic(ImgprocessingTestCase):

    def test_saveAndLoad(self):

        # test basic saving a loading functionality
        # new registration methods should add tests
        # for loading and saving

        random.seed(42)
        ref = random.randn(25, 25)

        im = shift(ref, [2, 0], mode='constant', order=0)
        im2 = shift(ref, [0, 2], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays([im, im2])
        reg = Registration('crosscorr')
        reg.prepare(ref)
        model1 = reg.fit(imIn)

        t = tempfile.mkdtemp()
        model1.save(t + '/test.json')
        # with open(t + '/test.json', 'r') as fp:
        #    print fp.read()
        model2 = Registration.load(t + '/test.json')
        # print model2

        out1 = model1.transform(imIn).first()[1]
        out2 = model2.transform(imIn).first()[1]

        assert_true(allclose(out1, out2))

    def test_run(self):

        # tests the run method which combines fit and transform

        random.seed(42)
        ref = random.randn(25, 25)

        im = shift(ref, [2, 0], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)

        reg = Registration('crosscorr')
        reg.prepare(ref)
        model = reg.fit(imIn)
        out1 = model.transform(imIn).first()[1]
        out2 = reg.run(imIn).first()[1]

        assert_true(allclose(out1, out2))

    def test_reference2d(self):
        """test default reference calculation in 2D
        """
        random.seed(42)

        im0 = random.rand(25, 25).astype('float')
        im1 = random.rand(25, 25).astype('float')
        im2 = random.rand(25, 25).astype('float')
        imIn = ImagesLoader(self.sc).fromArrays([im0, im1, im2])

        reg = Registration('crosscorr').prepare(imIn)
        assert_true(allclose(reg.reference, (im0 + im1 + im2) / 3))

        reg = Registration('crosscorr').prepare(imIn, startIdx=0, stopIdx=2)
        assert_true(allclose(reg.reference, (im0 + im1) / 2))

        reg = Registration('crosscorr').prepare(imIn, startIdx=1, stopIdx=2)
        assert_true(allclose(reg.reference, im1))

        reg = Registration('crosscorr').prepare(imIn, defaultNImages=1)
        assert_true(allclose(reg.reference, im1))

        imgs = [random.randn(25, 25).astype('float') for _ in xrange(27)]
        imIn = ImagesLoader(self.sc).fromArrays(imgs)
        reg = Registration('crosscorr').prepare(imIn)
        expected = mean(dstack(imgs[3:23]), axis=2)
        assert_true(allclose(expected, reg.reference))

    def test_reference_3d(self):
        """ test default reference calculation in 3D
        """
        random.seed(42)
        im0 = random.randn(25, 25, 3).astype('float')
        im1 = random.randn(25, 25, 3).astype('float')
        imIn = ImagesLoader(self.sc).fromArrays([im0, im1])
        reg = Registration('crosscorr').prepare(imIn)
        assert_true(allclose(reg.reference, (im0 + im1) / 2))


class TestCrossCorr(ImgprocessingTestCase):

    def test_crosscorrImage(self):
        random.seed(42)
        ref = random.randn(25, 25)

        reg = Registration('crosscorr')

        im = shift(ref, [2, 0], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.prepare(ref).fit(imIn).transformations[0].delta
        imOut = reg.prepare(ref).run(imIn).first()[1]
        assert_true(allclose(ref[:-2, :], imOut[:-2, :]))
        assert_true(allclose(paramOut, [2, 0]))

        im = shift(ref, [0, 2], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.prepare(ref).fit(imIn).transformations[0].delta
        imOut = reg.prepare(ref).run(imIn).first()[1]
        assert_true(allclose(ref[:, :-2], imOut[:, :-2]))
        assert_true(allclose(paramOut, [0, 2]))

        im = shift(ref, [2, -2], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.prepare(ref).fit(imIn).transformations[0].delta
        imOut = reg.prepare(ref).run(imIn).first()[1]
        assert_true(allclose(ref[:-2, 2:], imOut[:-2, 2:]))
        assert_true(allclose(paramOut, [2, -2]))

        im = shift(ref, [-2, 2], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.prepare(ref).fit(imIn).transformations[0].delta
        imOut = reg.prepare(ref).run(imIn).first()[1]
        assert_true(allclose(ref[2:, :-2], imOut[2:, :-2]))
        assert_true(allclose(paramOut, [-2, 2]))

    def test_toarray(self):
        random.seed(42)
        ref = random.randn(25, 25)

        reg = Registration('crosscorr')

        im = shift(ref, [2, 0], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)
        paramOut = reg.prepare(ref).fit(imIn).toArray()
        assert_true(isinstance(paramOut, ndarray))
        assert_true(allclose(paramOut, [[2, 0]]))

    def test_crosscorrVolume(self):

        random.seed(42)
        ref = random.randn(25, 25, 3)
        im = shift(ref, [2, -2, 0], mode='constant', order=0)
        imIn = ImagesLoader(self.sc).fromArrays(im)

        # use 3D cross correlation
        paramOut = Registration('crosscorr').prepare(ref).fit(imIn).transformations[0].delta
        imOut = Registration('crosscorr').prepare(ref).run(imIn).first()[1]
        assert_true(allclose(paramOut, [2, -2, 0]))
        assert_true(allclose(ref[:-2, 2:, :], imOut[:-2, 2:, :]))

        # use 2D cross correlation on each plane
        paramOut = Registration('planarcrosscorr').prepare(ref).fit(imIn).transformations[0].delta
        imOut = Registration('planarcrosscorr').prepare(ref).run(imIn).first()[1]
        assert_true(allclose(paramOut, [[2, -2], [2, -2], [2, -2]]))
        assert_true(allclose(ref[:-2, 2:, :], imOut[:-2, 2:, :]))


class TestTransformation(ImgprocessingTestCase):
    def test_translation(self):
        im = zeros((5, 5))
        im[2:4, 2:4] = 1.0
        expected = zeros((5, 5))
        expected[3:, :2] = 1.0
        tfm = TranslationTransformation(shift=(1, -2))
        out = tfm.apply(im)
        assert_true(allclose(out, expected))

    def _run_euclidean_test(self, sz):
        im = zeros((sz, sz))
        im[:1, :] = 1.0
        expected = zeros((sz, sz))
        expected[:, 2] = 1.0
        tfm = EuclideanTransformation(shift=(2.0, 0.0), rotation=pi/2)
        out = tfm.apply(im)
        assert_true(allclose(out, expected))

    def test_euclidean(self):
        self._run_euclidean_test(5)
        self._run_euclidean_test(6)


class TestLucasKanade(ImgprocessingTestCase):
    def test_euclidean(self):
        vol = zeros((16, 16))
        vol[4:12, 4:8] = 1.0
        rot = pi / 8.0
        trueTfm = EuclideanTransformation((0, 1),  rot)
        volTfm = trueTfm.apply(vol)

        reg = Registration('lucaskanade', transformationType='Euclidean').prepare(volTfm)
        predTfm = reg.getTransform(vol)
        print trueTfm.getParams(), predTfm.getParams()
        assert_true(allclose(trueTfm.getParams(), predTfm.getParams(), atol=1e-2))

    def test_translation(self):
        vol = zeros((16, 16))
        vol[4:12, 4:8] = 1.0
        trueTfm = TranslationTransformation((1.75, -2.5))
        volTfm = trueTfm.apply(vol)
        reg = Registration('lucaskanade', transformationType='Translation').prepare(volTfm)
        predTfm = reg.getTransform(vol)
        print trueTfm.getParams(), predTfm.getParams()
        assert_true(allclose(trueTfm.getParams(), predTfm.getParams(), atol=1e-2))
