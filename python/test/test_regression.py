import shutil
import tempfile
from numpy import array, allclose, pi, r_, mean, std, zeros, sqrt
from thunder import LinearRegression
from thunder import TuningModel
from test_utils import PySparkTestCase
from thunder.rdds.series import Series
from sklearn import linear_model as lm


class TestLinearRegression(PySparkTestCase):
    """Test accuracy of linear and bilinear regression
    models by building small design matrices and testing
    on small data against ground truth
    (ground truth derived by doing the algebra in MATLAB)

    """
    def setUp(self):
        super(TestLinearRegression, self).setUp()
        self.X = array([ [-0.4309741,   0.43440693,  0.19946369,  1.40428728],
                         [ 0.54587086, -1.1092286,  -0.27258427,  0.35205421],
                         [-0.4432777,   0.40580108,  0.20938645,  0.26480389],
                         [-0.53239659, -0.90966912, -0.13967252,  1.38274305],
                         [ 0.35731376,  0.39878607,  0.07762888,  1.82299252],
                         [ 0.36687294, -0.17079843, -0.17765573,  0.87161138],
                         [ 0.3017848,   1.36537541,  0.91211512, -0.80570055],
                         [-0.72330999,  0.36319617,  0.08986615, -0.7830115 ],
                         [ 1.11477831,  0.41631623,  0.11104172, -0.90049209],
                         [-1.62162968,  0.46928843,  0.62996118,  1.08668594]])
        self.y0 = array([ 4.57058016, -4.06400691,  4.25957933,  2.01583617,  0.34791879,
                         -0.9113852, 3.41167194,  5.26059279, -2.35116878,  6.28263909])
        self.y = Series(self.sc.parallelize([ ( (1,), self.y0 ) ]))
        self.tol = 1E-3

    def test_ordinaryLinearRegression(self):
        #sklearn's normalize=True option 'unscales' after fitting, we do not, so we do the scaling for them
        Xscaled = (self.X - mean(self.X, axis=0))/std(self.X, axis=0, ddof=1) 

        # no intercept, no scaling
        betas = LinearRegression(intercept=False).fit(self.X, self.y).coeffs.values().first()
        betas0 = lm.LinearRegression(fit_intercept=False).fit(self.X, self.y0).coef_
        assert(allclose(betas, betas0, atol=self.tol))

        # intercept, no scaling
        betas = LinearRegression().fit(self.X, self.y).coeffs.values().first()
        result = lm.LinearRegression(fit_intercept=True).fit(self.X, self.y0)
        betas0 = r_[result.intercept_, result.coef_]
        assert(allclose(betas, betas0, atol=self.tol))

        # no intercept, scaling
        betas = LinearRegression(intercept=False, zscore=True).fit(self.X, self.y).coeffs.values().first()
        betas0 = lm.LinearRegression(fit_intercept=False).fit(Xscaled, self.y0).coef_
        assert(allclose(betas, betas0, atol=self.tol))

        # intercept, scaling
        betas = LinearRegression(zscore=True).fit(self.X, self.y).coeffs.values().first()
        result = lm.LinearRegression().fit(Xscaled, self.y0)
        betas0 = r_[result.intercept_, result.coef_]
        assert(allclose(betas, betas0, atol=self.tol))

    def test_tikhonovLinearRegression(self):
        R = array([[1, -2, 1, 0],
                   [0, 1, -2, 0]], dtype='float')
        c = 2.0
        
        # no intercept, no normalization
        # tikhonov regularization can be recast in the form of OLS regression with an augmented design matrix
        Xaug = r_[self.X, sqrt(c)*R]
        yaug = r_[self.y0, zeros(R.shape[0])]

        betas = LinearRegression('tikhonov', intercept=False,  R=R, c=c).fit(self.X, self.y).coeffs.values().first()
        betas0 = lm.LinearRegression(fit_intercept=False).fit(Xaug, yaug).coef_
        assert(allclose(betas, betas0, atol=self.tol))

        # intercept, no normalization
        Xscaled = (self.X - mean(self.X, axis=0))/std(self.X, axis=0, ddof=1)
        Xaug = r_[Xscaled, sqrt(c)*R] 
        yint = mean(self.y0)
        yaug = r_[self.y0-yint, zeros(R.shape[0])] 

        betas = LinearRegression('tikhonov', zscore=True, R=R, c=c).fit(self.X, self.y).coeffs.values().first()
        result = lm.LinearRegression(fit_intercept=False).fit(Xaug, yaug)
        betas0 = r_[yint, result.coef_]
        assert(allclose(betas, betas0, atol=self.tol))

    def test_RidgeLinearRegression(self):
        c = 2.0
        betas = LinearRegression('ridge', intercept=False, c=c).fit(self.X, self.y).coeffs.values().first()
        betas0 = lm.Ridge(fit_intercept=False, alpha=c).fit(self.X, self.y0).coef_
        assert(allclose(betas, betas0, atol=self.tol))

    def test_LinearRegressionModelMethods(self):
        model = LinearRegression().fit(self.X, self.y)
        model0 = lm.LinearRegression().fit(self.X, self.y0)

        yhat = model.predict(self.X).values().first()
        yhat0 = model0.predict(self.X)
        assert(allclose(yhat, yhat0, atol=self.tol))

        R2_initial = model.stats.values().first()
        R2_score = model.score(self.X, self.y).values().first()
        R2_0 = model0.score(self.X, self.y0)
        assert(allclose(R2_initial, R2_score, atol=self.tol))
        assert(allclose(R2_initial, R2_0, atol=self.tol))

        result1, result2 = model.predictAndScore(self.X, self.y)
        yhat = result1.values().first()
        R2 = result2.values().first()
        assert(allclose(yhat, yhat0, atol=self.tol))
        assert(allclose(R2, R2_0, atol=self.tol))

class TestNonlinearRegression(PySparkTestCase):
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
    def test_gaussianTuningModel(self):
        data = Series(self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))]))
        s = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        model = TuningModel.load(s, "gaussian")
        params = model.fit(data)
        tol = 1E-4  # to handle rounding errors
        assert(allclose(params.select('center').values().collect()[0], array([0.36262]), atol=tol))
        assert(allclose(params.select('spread').values().collect()[0], array([0.01836]), atol=tol))

    def test_circularTuningModel(self):
        data = Series(self.sc.parallelize([(1, array([1.5, 2.3, 6.2, 5.1, 3.4, 2.1]))]))
        s = array([-pi/2, -pi/3, -pi/4, pi/4, pi/3, pi/2])
        model = TuningModel.load(s, "circular")
        params = model.fit(data)
        tol = 1E-4  # to handle rounding errors
        assert(allclose(params.select('center').values().collect()[0], array([0.10692]), atol=tol))
        assert(allclose(params.select('spread').values().collect()[0], array([1.61944]), atol=tol))
