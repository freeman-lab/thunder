import shutil
import tempfile
from numpy import array, vstack
from numpy.testing import assert_array_almost_equal
from scipy.stats import ttest_ind
from thunder.classification import MassUnivariateClassifier
from test_utils import PySparkTestCase


class ClassificationTestCase(PySparkTestCase):
    def setUp(self):
        super(ClassificationTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClassificationTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestMassUnivariateClassification(ClassificationTestCase):
    """Test accuracy of mass univariate classification on small
    test data sets with either 1 or 2 features
    """

    def test_mass_univariate_classification_ttest_1d(self):
        """Simple classification problem, 1d features"""
        X = array([-1, -0.1, -0.1, 1, 1, 1.1])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels)])

        clf = MassUnivariateClassifier.load(params, "ttest")

        # should match direct calculation using scipy
        data = self.sc.parallelize(zip([1], [X]))
        result = clf.classify(data).map(lambda (_, v): v).collect()
        ground_truth = ttest_ind(X[labels == 1], X[labels == 2])
        assert_array_almost_equal(result[0], ground_truth[0])

    def test_mass_univariate_classification_ttest_2d(self):
        """Simple classification problem, 2d features"""
        X = array([-1, -2, -0.1, -2, -0.1, -2.1, 1, 1.1, 1, 1, 1.1, 2])
        features = array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        samples = array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels), ('features', features), ('samples', samples)])

        clf = MassUnivariateClassifier.load(params, "ttest")

        # should match direct calculation using scipy

        # test first feature only
        data = self.sc.parallelize(zip([1], [X]))
        result = clf.classify(data, [[1]]).map(lambda (_, v): v).collect()
        ground_truth = ttest_ind(X[features == 1][:3], X[features == 1][3:])
        assert_array_almost_equal(result[0], ground_truth[0])

        # test both features
        result = clf.classify(data, [[1, 2]]).map(lambda (_, v): v).collect()
        ground_truth = ttest_ind(vstack((X[features == 1][:3], X[features == 2][:3])).T,
                                 vstack((X[features == 1][3:], X[features == 2][3:])).T)
        assert_array_almost_equal(result[0][0], ground_truth[0])

    def test_mass_univariate_classification_gnb_1d(self):
        """Simple classification problem, 1d features"""
        X1 = array([-1, -1, -1.2, 1, 1, 1.2])
        X2 = array([-1, -1, 1.2, 1, 1, 1.2])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels)])

        clf = MassUnivariateClassifier.load(params, "gaussnaivebayes", cv=0)

        # should predict perfectly
        data = self.sc.parallelize(zip([1], [X1]))
        result = clf.classify(data).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [1.0])

        # should predict all but one correctly
        data = self.sc.parallelize(zip([1], [X2]))
        result = clf.classify(data).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [5.0/6.0])

    def test_mass_univariate_classification_gnb_2d(self):
        """Simple classification problem, 2d features"""

        X = array([-1, 1, -2, -1, -3, -2, 1, 1, 2, 1, 3, 2])
        features = array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        samples = array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        labels = array([1, 1, 1, 2, 2, 2])
        params = dict([('labels', labels), ('features', features), ('samples', samples)])
        clf = MassUnivariateClassifier.load(params, "gaussnaivebayes", cv=0)

        data = self.sc.parallelize(zip([1], [X]))

        # first feature predicts perfectly
        result = clf.classify(data, [[1]]).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [1.0])

        # second feature gets one wrong
        result = clf.classify(data, [[2]]).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [5.0/6.0])

        # two features together predict perfectly
        result = clf.classify(data, [[1, 2]]).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [1.0])

        # test iteration over multiple feature sets
        result = clf.classify(data, [[1, 2], [2]]).map(lambda (_, v): v).collect()
        assert_array_almost_equal(result[0], [1.0, 5.0/6.0])



