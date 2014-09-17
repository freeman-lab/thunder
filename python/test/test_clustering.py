import shutil
import tempfile
from numpy import array, array_equal
from thunder.clustering.kmeans import KMeans, KMeansModel
from thunder.utils import DataSets
from test_utils import PySparkTestCase


class ClusteringTestCase(PySparkTestCase):
    def setUp(self):
        super(ClusteringTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(ClusteringTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestKMeans(ClusteringTestCase):
    def test_kmeans_k1(self):
        """ With k=1 always get one cluster centered on the mean"""

        data_local = [
            array([1.0, 2.0, 6.0]),
            array([1.0, 3.0, 0.0]),
            array([1.0, 4.0, 6.0])]

        data = self.sc.parallelize(zip(range(1, 4), data_local))

        model = KMeans(k=1, maxiter=20, tol=0.001).train(data)
        labels = model.predict(data)
        assert array_equal(model.centers[0], array([1.0, 3.0, 4.0]))
        assert array_equal(labels.map(lambda (_, v): v).collect(), array([0, 0, 0]))

    def test_kmeans_k2(self):
        """ Test k=2 also with more points"""

        data, centerstrue = DataSets.make(self.sc, "kmeans",
                                          k=2, nrecords=50, npartitions=5, seed=42, returnparams=True)
        centerstrue = KMeansModel(centerstrue)

        model = KMeans(k=2, maxiter=20, tol=0.001, init="sample").train(data)

        labels = array(model.predict(data).values().collect())
        labelstrue = array(centerstrue.predict(data).values().collect())
        print(labels)
        print(labelstrue)

        assert(array_equal(labels, labelstrue) or array_equal(labels, 1 - labelstrue))



