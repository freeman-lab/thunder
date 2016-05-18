import pytest
from numpy import arange, array, allclose, ones, float64, asarray

from thunder.images.readers import fromlist

pytestmark = pytest.mark.usefixtures("eng")


def test_conversion(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=eng)
    vals = data.toblocks((2, 2)).collect_blocks()
    truth = [array([a[0:2, 0:2], a[0:2, 0:2]]), array([a[2:4, 0:2], a[2:4, 0:2]])]
    assert allclose(vals, truth)


def test_full(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=eng)
    vals = data.toblocks((4,2)).collect_blocks()
    truth = [a, a]
    assert allclose(vals, truth)


def test_padding(eng):
    a = arange(30).reshape((5, 6))
    data = fromlist([a, a], engine=eng)

    blocks = data.toblocks((2, 3), padding=(1, 1))
    vals = blocks.collect_blocks()
    shapes = list(map(lambda x: x.shape, vals))
    truth = [(2, 3, 4), (2, 3, 4), (2, 4, 4), (2, 4, 4), (2, 2, 4), (2, 2, 4)]
    assert allclose(array(shapes), array(truth))

    truth = data.toarray()
    assert allclose(data.toblocks((2, 3), padding=1).toarray(), truth)
    assert allclose(data.toblocks((2, 3), padding=(0, 1)).toarray(), truth)
    assert allclose(data.toblocks((2, 3), padding=(1, 1)).toarray(), truth)


def test_count(eng):
    a = arange(8).reshape((2, 4))
    data = fromlist([a], engine=eng)
    assert data.toblocks((1, 1)).count() == 8
    assert data.toblocks((1, 2)).count() == 4
    assert data.toblocks((2, 2)).count() == 2
    assert data.toblocks((2, 4)).count() == 1


def test_conversion_series(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a], engine=eng)
    vals = data.toblocks((1, 2)).toseries().toarray()
    assert allclose(vals, a)


def test_conversion_series_3d(eng):
    a = arange(24).reshape((2, 3, 4))
    data = fromlist([a], engine=eng)
    vals = data.toblocks((2, 3, 4)).toseries().toarray()
    assert allclose(vals, a)


def test_roundtrip(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=eng)
    vals = data.toblocks((2, 2)).toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_series_roundtrip_simple(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=eng)
    vals = data.toseries().toimages()
    assert allclose(vals.toarray(), data.toarray())


def test_shape(eng):
    data = fromlist([ones((30, 30)) for _ in range(0, 3)], engine=eng)
    blocks = data.toblocks((10, 10))
    values = blocks.collect_blocks()
    assert blocks.blockshape == (3, 10, 10)
    assert all([v.shape == (3, 10, 10) for v in values])


def test_map(eng):
    a = arange(8).reshape((4, 2))
    data = fromlist([a, a], engine=eng)
    map1 = data.toblocks((4, 2)).map(lambda x: 1.0 * x, dtype=float64).toimages()
    map2 = data.toblocks((4, 2)).map(lambda x: 1.0 * x).toimages()
    assert map1.dtype == float64
    assert map2.dtype == float64

def test_map_generic(eng):
    a = arange(3*4).reshape((3, 4))
    data = fromlist([a, a], engine=eng)
    b = asarray(data.toblocks((2, 2)).map_generic(lambda x: [0, 1]))
    assert b.shape == (2, 2)
    assert b.dtype == object
    truth = [v == [0, 1] for v in b.flatten()]
    assert all(truth)
