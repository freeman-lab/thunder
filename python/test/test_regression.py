import shutil
import tempfile
from numpy import array, allclose, pi
from thunder.regression import RegressionModel, TuningModel
from test_utils import PySparkTestCase


class RegressionTestCase(PySparkTestCase):
    def setUp(self):
        super(RegressionTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(RegressionTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestRegress(RegressionTestCase):
    """Test accuracy of linear and bilinear regression
    models by building small design matrices and testing
    on small data against ground truth
    (ground truth derived by doing the algebra in MATLAB)

    """
    def test_linear_regress(self):
        data = self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))])
        x = array([
            array([1, 0, 0, 0, 0, 0]),
            array([0, 1, 0, 0, 0, 0])
        ])
        model = RegressionModel.load(x, "linear")
        betas, stats, resid = model.fit(data)
        assert(allclose(betas.map(lambda (_, v): v).collect()[0], array([-2.7, -1.9])))
        assert(allclose(stats.map(lambda (_, v): v).collect()[0], array([0.42785299])))
        assert(allclose(resid.map(lambda (_, v): v).collect()[0], array([0, 0, 2, 0.9, -0.8, -2.1])))

    def test_blinear_regress(self):
        data = self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))])
        x1 = array([
            array([1, 0, 1, 0, 1, 0]),
            array([0, 1, 0, 1, 0, 1])
        ])
        x2 = array([
            array([1, 1, 0, 0, 0, 0]),
            array([0, 0, 1, 1, 0, 0]),
            array([0, 0, 0, 0, 1, 1])
        ])
        model = RegressionModel.load((x1, x2), "bilinear")
        betas, stats, resid = model.fit(data)
        tol = 1E-4  # to handle rounding errors
        assert(allclose(betas.map(lambda (_, v): v).collect()[0], array([-3.1249, 5.6875, 0.4375]), atol=tol))
        assert(allclose(stats.map(lambda (_, v): v).collect()[0], array([0.6735]), tol))
        assert(allclose(resid.map(lambda (_, v): v).collect()[0], array([0, -0.8666, 0, 1.9333, 0, -1.0666]), atol=tol))


class TestTuning(RegressionTestCase):
    """Test accuracy of gaussian and circular tuning
    by building small stimulus arrays and testing
    on small data against ground truth
    (ground truth for gaussian tuning
    derived by doing the algebra in MATLAB,
    ground truth for circular tuning
    derived from MATLAB's circular statistics toolbox
    circ_mean and circ_kappa functions)

    Also tests that main analysis script runs without crashing
    (separately, to test a variety of inputs)
    """
    def test_gaussian_tuning_model(self):
        data = self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))])
        s = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        model = TuningModel.load(s, "gaussian")
        params = model.fit(data)
        tol = 1E-4  # to handle rounding errors
        assert(allclose(params.map(lambda (_, v): v).collect()[0], array([0.36262, 0.01836]), atol=tol))

    def test_circular_tuning_model(self):
        data = self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))])
        s = array([-pi/2, -pi/3, -pi/4, pi/4, pi/3, pi/2])
        model = TuningModel.load(s, "circular")
        params = model.fit(data)
        tol = 1E-4  # to handle rounding errors
        assert(allclose(params.map(lambda (_, v): v).collect()[0], array([0.10692, 1.61944]), atol=tol))
