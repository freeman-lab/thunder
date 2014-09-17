from collections import namedtuple
from nose.tools import assert_equals
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


def test_subtoind_parameterized():
    SubToIndParameters = namedtuple('SubToIndParameters', ['subscripts', 'dims', 'indices', 'order', 'onebased'])
    parameters = [SubToIndParameters([(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                      (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)],
                                     dims=(2, 3, 2), indices=range(1, 13), order='F', onebased=True,),
                  SubToIndParameters([(0, 1, 1)], dims=(2, 3, 2), indices=[0], order='F', onebased=True,),
                  SubToIndParameters([(-1, 1), (0, 1), (1, 1), (2, 1), (3, 1)], dims=(2, 1),
                                     indices=[-1, 0, 1, 2, 3], order='F', onebased=True,),
                  SubToIndParameters([(-1,), (0,), (1,), (2,), (3,)], dims=(1,),
                                     indices=[-1, 0, 1, 2, 3], order='F', onebased=True,),
                  SubToIndParameters([(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (1, 3, 1), (1, 3, 2),
                                      (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2), (2, 3, 1), (2, 3, 2)],
                                     dims=(2, 3, 2), indices=range(1, 13), order='C', onebased=True,),
                  SubToIndParameters([(1, 1, 0)], dims=(2, 3, 2), indices=[0], order='C', onebased=True,),
                  SubToIndParameters([(1, -1), (1, 0), (1, 1), (1, 2), (1, 3)], dims=(2, 1),
                                     indices=[-1, 0, 1, 2, 3], order='C', onebased=True,),
                  SubToIndParameters([(-1,), (0,), (1,), (2,), (3,)], dims=(1,),
                                     indices=[-1, 0, 1, 2, 3], order='C', onebased=True,),

                  SubToIndParameters([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0),
                                      (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)],
                                     dims=(2, 3, 2), indices=range(12), order='F', onebased=False,),
                  SubToIndParameters([(-1, 0, 0)], dims=(2, 3, 2), indices=[-1], order='F', onebased=False,),
                  SubToIndParameters([(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)], dims=(2, 1),
                                     indices=[-2, -1, 0, 1, 2], order='F', onebased=False,),
                  SubToIndParameters([(-2,), (-1,), (0,), (1,), (2,)], dims=(1,),
                                     indices=[-2, -1, 0, 1, 2], order='F', onebased=False,),
                  SubToIndParameters([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 2, 0), (0, 2, 1),
                                      (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)],
                                     dims=(2, 3, 2), indices=range(12), order='C', onebased=False,),
                  SubToIndParameters([(0, 0, -1)], dims=(2, 3, 2), indices=[-1], order='C', onebased=False,),
                  SubToIndParameters([(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)], dims=(2, 1),
                                     indices=[-2, -1, 0, 1, 2], order='C', onebased=False,),
                  SubToIndParameters([(-2,), (-1,), (0,), (1,), (2,)], dims=(1,),
                                     indices=[-2, -1, 0, 1, 2], order='C', onebased=False,)
                  ]

    def check_subtoind_result(si_param):
        # attach dummy value 'x' to subscripts to match expected input to subtoind
        data = map(lambda d: (d, 'x'), si_param.subscripts)
        results = subtoind(data, si_param.dims, order=si_param.order, onebased=si_param.onebased)
        # check results individually to highlight specific failures
        for res, expected, subscript in zip(results, si_param.indices, si_param.subscripts):
            assert_equals(expected, res[0], 'Got index %d instead of %d for subscript:%s, dims:%s' %
                          (res[0], expected, str(subscript), str(si_param.dims)))

    for param in parameters:
        yield check_subtoind_result, param


def test_indtosub_parameterized():
    IndToSubParameters = namedtuple('IndToSubParameters', ['indices', 'dims', 'subscripts', 'order', 'onebased'])
    parameters = [IndToSubParameters(range(1, 13), dims=(2, 3, 2), order='F', onebased=True,
                                     subscripts=[(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                                 (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]),
                  # indices out of range are wrapped back into range with >1 dimension:
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1, 2), order='F', onebased=True,
                                     subscripts=[(1, 1), (1, 2), (1, 1), (1, 2), (1, 1)]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1, 2), order='C', onebased=True,
                                     subscripts=[(1, 1), (1, 2), (1, 1), (1, 2), (1, 1)]),
                  # note with only one dimension, we no longer wrap, and no longer return tuples:
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1,), order='F', onebased=True,
                                     subscripts=[-1, 0, 1, 2, 3]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1,), order='C', onebased=True,
                                     subscripts=[-1, 0, 1, 2, 3]),
                  IndToSubParameters(range(1, 6), dims=(2, 3), order='F', onebased=True,
                                     subscripts=[(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)]),
                  IndToSubParameters(range(1, 6), dims=(2, 3), order='C', onebased=True,
                                     subscripts=[(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]),
                  IndToSubParameters(range(1, 13), dims=(2, 3, 4), order='F', onebased=True,
                                     subscripts=[(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (1, 3, 1), (2, 3, 1),
                                                 (1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2), (1, 3, 2), (2, 3, 2)]),
                  IndToSubParameters(range(1, 13), dims=(2, 3, 4), order='C', onebased=True,
                                     subscripts=[(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4),
                                                 (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4),
                                                 (1, 3, 1), (1, 3, 2), (1, 3, 3), (1, 3, 4)]),
                  IndToSubParameters(range(12), dims=(2, 3, 2), order='F', onebased=False,
                                     subscripts=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0),
                                                 (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1, 2), order='F', onebased=False,
                                     subscripts=[(0, 1), (0, 0), (0, 1), (0, 0), (0, 1)]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1, 2), order='C', onebased=False,
                                     subscripts=[(0, 1), (0, 0), (0, 1), (0, 0), (0, 1)]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1,), order='F', onebased=False,
                                     subscripts=[-1, 0, 1, 2, 3]),
                  IndToSubParameters([-1, 0, 1, 2, 3], dims=(1,), order='C', onebased=False,
                                     subscripts=[-1, 0, 1, 2, 3]),
                  IndToSubParameters(range(5), dims=(2, 3), order='F', onebased=False,
                                     subscripts=[(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]),
                  IndToSubParameters(range(5), dims=(2, 3), order='C', onebased=False,
                                     subscripts=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
                  IndToSubParameters(range(12), dims=(2, 3, 4), order='F', onebased=False,
                                     subscripts=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0), (1, 2, 0),
                                                 (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1), (1, 2, 1)]),
                  IndToSubParameters(range(12), dims=(2, 3, 4), order='C', onebased=False,
                                     subscripts=[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                                                 (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                                                 (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3)]),
                  ]

    def check_indtosub_result(indsub_param):
        # attach dummy value 'x' to indices to match expected input to indtosub
        data = map(lambda d: (d, 'x'), indsub_param.indices)
        results = indtosub(data, indsub_param.dims, order=indsub_param.order, onebased=indsub_param.onebased)
        for res, expected, index in zip(results, indsub_param.subscripts, indsub_param.indices):
            assert_equals(expected, res[0], 'Got subscript %s instead of %s for index:%d, dims:%s' %
                          (res[0], expected, index, str(indsub_param.dims)))

    for param in parameters:
        yield check_indtosub_result, param
