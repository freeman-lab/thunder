import shutil
import tempfile
from numpy import array, allclose
from thunder.utils import subtoind, indtosub, getdims
from test_utils import PySparkTestCase


class LoadTestCase(PySparkTestCase):
    def setUp(self):
        super(LoadTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(LoadTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)


class TestSubToInd(LoadTestCase):
    """Test conversion between linear and subscript indexing"""

    def test_sub_to_ind_rdd(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)

        data = self.sc.parallelize(data_local)
        dims = [2, 3, 2]
        inds = subtoind(data, dims).map(lambda (k, _): k).collect()
        assert(allclose(inds, array(range(1, 13))))

    def test_ind_to_sub_rdd(self):
        data_local = map(lambda x: (x, array([1.0])), range(1, 13))

        data = self.sc.parallelize(data_local)
        dims = [2, 3, 2]
        subs = indtosub(data, dims).map(lambda (k, _): k).collect()
        assert(allclose(subs, array([(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                     (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)])))

    def test_sub_to_ind_array(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)
        dims = [2, 3, 2]
        inds = map(lambda x: x[0], subtoind(data_local, dims))
        assert(allclose(inds, array(range(1, 13))))

    def test_ind_to_sub_array(self):
        data_local = map(lambda x: (x, array([1.0])), range(1, 13))
        dims = [2, 3, 2]
        subs = map(lambda x: x[0], indtosub(data_local, dims))
        assert(allclose(subs, array([(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                     (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)])))


class TestGetDims(LoadTestCase):
    """Test getting dimensions"""

    def test_get_dims_rdd(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)
        data = self.sc.parallelize(data_local)
        dims = getdims(data)
        assert(allclose(dims.max, (2, 3, 2)))
        assert(allclose(dims.count(), (2, 3, 2)))
        assert(allclose(dims.min, (1, 1, 1)))

    def test_get_dims_array(self):
        subs = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]
        data_local = map(lambda x: (x, array([1.0])), subs)
        dims = getdims(data_local)
        assert(allclose(dims.max, (2, 3, 2)))
        assert(allclose(dims.count(), (2, 3, 2)))
        assert(allclose(dims.min, (1, 1, 1)))
