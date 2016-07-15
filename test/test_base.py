import pytest
from numpy import allclose, array, asarray, add, ndarray, generic

from thunder import series, images

pytestmark = pytest.mark.usefixtures("eng")


def test_first(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    assert allclose(data.first(), [1, 2, 3])
    data = images.fromlist([array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])], engine=eng)
    assert allclose(data.first(), [[1, 2], [3, 4]])


def test_asarray(eng):
     data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
     converted = asarray(data)
     assert allclose(converted, [[1, 2, 3], [4, 5, 6]])


def test_casting(eng):
    data = series.fromlist([array([1, 2, 3], 'int16')], engine=eng)
    assert data.astype('int64').toarray().dtype == 'int64'
    assert data.astype('float32').toarray().dtype == 'float32'
    assert data.astype('float64').toarray().dtype == 'float64'
    assert data.astype('float16', casting='unsafe').toarray().dtype == 'float16'


def test_slicing(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    assert data.toarray().shape == (2, 3)
    assert data[:, :].shape == (2, 3)
    assert data[:, :].toarray().shape == (2, 3)
    assert data[0, :].shape == (1, 3)
    assert data[0, :].toarray().shape == (3,)


def test_toarray(eng):
    original = [array([1, 2, 3]), array([4, 5, 6])]
    data = series.fromlist(original, engine=eng)
    assert allclose(data.toarray(), original)
    original = [array([[1, 2], [3, 4]]), array([[5, 6], [7, 8]])]
    data = images.fromlist(original, engine=eng)
    assert allclose(data.toarray(), original)


def test_elementwise(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2raw = asarray([[7, 8, 9], [10, 11, 12]])
    mat1 = series.fromlist(mat1raw, engine=eng)
    mat2 = series.fromlist(mat2raw, engine=eng)
    result = mat1.element_wise(mat2, add)
    truth = mat1raw + mat2raw
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))


def test_elementwise_scalar(eng):
    matraw = asarray([[1, 2, 3], [4, 5, 6]])
    mat = series.fromlist(matraw, engine=eng)
    result = mat.element_wise(2, add)
    truth = matraw + 2
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))


def test_elementwise_plus(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2raw = asarray([[7, 8, 9], [10, 11, 12]])
    mat1 = series.fromlist(mat1raw, engine=eng)
    mat2 = series.fromlist(mat2raw, engine=eng)
    result = mat1.plus(mat2)
    truth = mat1raw + mat2raw
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(3))


def test_reduce(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    reduced = data.reduce(lambda x, y: x + y)
    assert allclose(reduced.shape, [1, 3])
    assert allclose(reduced.toarray(), [5, 7, 9])


def test_map(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    mapped = data.map(lambda x: x.sum())
    assert allclose(mapped.shape, [2, 1])
    assert allclose(mapped.toarray(), [6, 15])
    mapped = data.map(lambda x: x + 1)
    assert allclose(mapped.shape, [2, 3])
    assert allclose(mapped.toarray(), [[2, 3, 4], [5, 6, 7]])


def test_map_with_keys(eng):
    data = series.fromlist([array([1, 2, 3]), array([4, 5, 6])], engine=eng)
    mapped = data.map(lambda kv: kv[0] + kv[1], with_keys=True)
    assert allclose(mapped.shape, [2, 3])
    assert allclose(mapped.toarray(), [[1, 2, 3], [5, 6, 7]])
    data = images.fromlist([array([[1, 1], [1, 1]]), array([[2, 2], [2, 2]])], engine=eng)
    mapped = data.map(lambda kv: kv[0] + kv[1], with_keys=True)
    assert allclose(mapped.shape, [2, 2, 2])
    assert allclose(mapped.toarray(), [[[1, 1], [1, 1]], [[3, 3], [3, 3]]])


def test_repartition(eng):
    if eng is not None:
        data = images.fromlist([array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4]),
                                array([5, 5]), array([6, 6]), array([7, 7]), array([8, 8]),
                                array([9, 9]), array([10, 10]), array([11, 11]), array([12, 12])],
                               engine=eng, npartitions=10)
        assert allclose(data.first(), array([1, 1]))
        assert isinstance(data.first(), (ndarray, generic))
        data = data.repartition(3)
        assert allclose(data.first(), array([1, 1]))

        data = series.fromlist([array([1, 1]), array([2, 2]), array([3, 3]), array([4, 4]),
                                array([5, 5]), array([6, 6]), array([7, 7]), array([8, 8]),
                                array([9, 9]), array([10, 10]), array([11, 11]), array([12, 12])],
                               engine=eng, npartitions=10)
        assert allclose(data.first(), array([1, 1]))
        data = data.repartition(3)
        assert allclose(data.first(), array([1, 1]))
        assert isinstance(data.first(), (ndarray, generic))
