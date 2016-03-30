import pytest
from numpy import allclose, arange, array, asarray, dot, cov

from thunder.series.readers import fromlist, fromarray
from thunder.series.timeseries import TimeSeries

pytestmark = pytest.mark.usefixtures("eng")


def test_totimeseries(eng):
    data = fromlist([array([4, 5, 6, 7]), array([8, 9, 10, 11])], engine=eng)
    ts = data.totimeseries()
    assert isinstance(ts, TimeSeries)


def test_map(eng):
    data = fromlist([array([1, 2]), array([3, 4])], engine=eng)
    assert allclose(data.map(lambda x: x + 1).toarray(), [[2, 3], [4, 5]])


def test_map_singletons(eng):
    data = fromlist([array([4, 5, 6, 7]), array([8, 9, 10, 11])], engine=eng)
    mapped = data.map(lambda x: x.mean())
    assert mapped.shape == (2, 1)


def test_filter(eng):
    data = fromlist([array([1, 2]), array([3, 4])], engine=eng)
    assert allclose(data.filter(lambda x: x.sum() > 3).toarray(), [3, 4])


def test_sample(eng):
    data = fromlist([array([1, 5]), array([1, 10]), array([1, 15])], engine=eng)
    assert allclose(data.sample(3).shape, (3, 2))
    assert allclose(data.filter(lambda x: x.max() > 10).sample(1).toarray(), [1, 15])


def test_between(eng):
    data = fromlist([array([4, 5, 6, 7]), array([8, 9, 10, 11])], engine=eng)
    val = data.between(0, 2)
    assert allclose(val.index, array([0, 1]))
    assert allclose(val.toarray(), array([[4, 5], [8, 9]]))

def test_flatten(eng):
    arr = arange(2*2*5).reshape(2, 2, 5)
    data = fromarray(arr, engine=eng)
    assert data.flatten().shape == (4, 5)
    assert allclose(data.flatten().toarray(), arr.reshape(2*2, 5))


def test_first(eng):
    data = fromlist([array([4, 5, 6, 7]), array([8, 9, 10, 11])], engine=eng)
    assert allclose(data.first(), [4, 5, 6, 7])


def test_select(eng):
    index = ['label1', 'label2', 'label3', 'label4']
    data = fromlist([array([4, 5, 6, 7]), array([8, 9, 10, 11])], engine=eng, index=index)
    assert data.select('label1').shape == (2, 1)
    assert allclose(data.select('label1').toarray(), [4, 8])
    assert allclose(data.select(['label1']).toarray(), [4, 8])
    assert allclose(data.select(['label1', 'label2']).toarray(), array([[4, 5], [8, 9]]))
    assert data.select('label1').index == ['label1']
    assert data.select(['label1']).index == ['label1']


def test_series_stats(eng):
    data = fromlist([array([1, 2, 3, 4, 5])], engine=eng)
    assert allclose(data.series_mean().toarray(), 3.0)
    assert allclose(data.series_sum().toarray(), 15.0)
    assert allclose(data.series_median().toarray(), 3.0)
    assert allclose(data.series_std().toarray(), 1.4142135)
    assert allclose(data.series_stat('mean').toarray(), 3.0)
    assert allclose(data.series_stats().select('mean').toarray(), 3.0)
    assert allclose(data.series_stats().select('count').toarray(), 5)
    assert allclose(data.series_percentile(25).toarray(), 2.0)
    assert allclose(data.series_percentile((25, 75)).toarray(), array([2.0, 4.0]))


def test_standardize_axis1(eng):
    data = fromlist([array([1, 2, 3, 4, 5])], engine=eng)
    centered = data.center(1)
    standardized = data.standardize(1)
    zscored = data.zscore(1)
    assert allclose(centered.toarray(), array([-2, -1, 0, 1, 2]), atol=1e-3)
    assert allclose(standardized.toarray(),
                    array([0.70710,  1.41421,  2.12132,  2.82842,  3.53553]), atol=1e-3)
    assert allclose(zscored.toarray(),
                    array([-1.41421, -0.70710,  0,  0.70710,  1.41421]), atol=1e-3)


def test_standardize_axis0(eng):
    data = fromlist([array([1, 2]), array([3, 4])], engine=eng)
    centered = data.center(0)
    standardized = data.standardize(0)
    zscored = data.zscore(0)
    assert allclose(centered.toarray(), array([[-1, -1], [1, 1]]), atol=1e-3)
    assert allclose(standardized.toarray(), array([[1, 2], [3, 4]]), atol=1e-3)
    assert allclose(zscored.toarray(), array([[-1, -1], [1, 1]]), atol=1e-3)


def test_squelch(eng):
    data = fromlist([array([1, 2]), array([3, 4])], engine=eng)
    squelched = data.squelch(5)
    assert allclose(squelched.toarray(), [[0, 0], [0, 0]])
    squelched = data.squelch(3)
    assert allclose(squelched.toarray(), [[0, 0], [3, 4]])
    squelched = data.squelch(1)
    assert allclose(squelched.toarray(), [[1, 2], [3, 4]])


def test_correlate(eng):
    data = fromlist([array([1, 2, 3, 4, 5])], engine=eng)
    sig = [4, 5, 6, 7, 8]
    corr = data.correlate(sig).toarray()
    assert allclose(corr, 1)
    sigs = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
    corrs = data.correlate(sigs).toarray()
    assert allclose(corrs, [1, -1])


def test_correlate_multiindex(eng):
    index = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    data = fromlist([array([1, 2, 3, 4, 5])], index=asarray(index).T, engine=eng)
    sig = [4, 5, 6, 7, 8]
    corr = data.correlate(sig).toarray()
    assert allclose(corr, 1)
    sigs = [[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]]
    corrs = data.correlate(sigs).toarray()
    assert allclose(corrs, [1, -1])


def test_clip(eng):
    data = fromlist([array([1, 2, 3, 4, 5])], engine=eng)
    assert allclose(data.clip(2).toarray(), [2, 2, 3, 4, 5])
    assert allclose(data.clip(2, 3).toarray(), [2, 2, 3, 3, 3])


def test_mean(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.mean().toarray()
    expected = data.toarray().mean(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_sum(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.sum().toarray()
    expected = data.toarray().sum(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'int64'


def test_var(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.var().toarray()
    expected = data.toarray().var(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_std(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.std().toarray()
    expected = data.toarray().std(axis=0)
    assert allclose(val, expected)
    assert str(val.dtype) == 'float64'


def test_max(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.max().toarray()
    expected = data.toarray().max(axis=0)
    assert allclose(val, expected)


def test_min(eng):
    data = fromlist([arange(8), arange(8)], engine=eng)
    val = data.min().toarray()
    expected = data.toarray().min(axis=0)
    assert allclose(val, expected)


def test_index_setting(eng):
    data = fromlist([array([1, 2, 3]), array([2, 2, 4]), array([4, 2, 1])], engine=eng)
    assert allclose(data.index, array([0, 1, 2]))
    data.index = [3, 2, 1]
    assert allclose(data.index, [3, 2, 1])
    with pytest.raises(ValueError):
        data.index = 5
    with pytest.raises(ValueError):
        data.index = [1, 2]


def test_select_by_index(eng):
    data = fromlist([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], engine=eng)
    result = data.select_by_index(1)
    assert allclose(result.toarray(), array([4, 5, 6, 7]))
    assert allclose(result.index, array([1, 1, 1, 1]))
    result = data.select_by_index(1, squeeze=True)
    assert allclose(result.index, array([0, 1, 2, 3]))
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data.index = array(index).T
    result, mask = data.select_by_index(0, level=2, return_mask=True)
    assert allclose(result.toarray(), array([0, 2, 6, 8]))
    assert allclose(result.index, array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]))
    assert allclose(mask, array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]))
    result = data.select_by_index(0, level=2, squeeze=True)
    assert allclose(result.toarray(), array([0, 2, 6, 8]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    result = data.select_by_index([1, 0], level=[0, 1])
    assert allclose(result.toarray(), array([6, 7]))
    assert allclose(result.index, array([[1, 0, 0], [1, 0, 1]]))
    result = data.select_by_index(val=[0, [2,3]], level=[0, 2])
    assert allclose(result.toarray(), array([4, 5]))
    assert allclose(result.index, array([[0, 1, 2], [0, 1, 3]]))
    result = data.select_by_index(1, level=1, filter=True)
    assert allclose(result.toarray(), array([0, 1, 6, 7]))
    assert allclose(result.index, array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]))


def test_aggregate_by_index(eng):
    data = fromlist([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], engine=eng)
    result = data.aggregate_by_index(sum)
    assert allclose(result.toarray(), array([6, 22, 38]))
    assert allclose(result.index, array([0, 1, 2]))
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data.index = array(index).T
    result = data.aggregate_by_index(sum, level=[0, 1])
    assert allclose(result.toarray(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))


def test_stat_by_index(eng):
    data = fromlist([arange(12)], index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], engine=eng)
    assert allclose(data.stat_by_index('sum').toarray(), array([6, 22, 38]))
    assert allclose(data.stat_by_index('mean').toarray(), array([1.5, 5.5, 9.5]))
    assert allclose(data.stat_by_index('min').toarray(), array([0, 4, 8]))
    assert allclose(data.stat_by_index('max').toarray(), array([3, 7, 11]))
    assert allclose(data.stat_by_index('count').toarray(), array([4, 4, 4]))
    assert allclose(data.stat_by_index('median').toarray(), array([1.5, 5.5, 9.5]))
    assert allclose(data.sum_by_index().toarray(), array([6, 22, 38]))
    assert allclose(data.mean_by_index().toarray(), array([1.5, 5.5, 9.5]))
    assert allclose(data.min_by_index().toarray(), array([0, 4, 8]))
    assert allclose(data.max_by_index().toarray(), array([3, 7, 11]))
    assert allclose(data.count_by_index().toarray(), array([4, 4, 4]))
    assert allclose(data.median_by_index().toarray(), array([1.5, 5.5, 9.5]))


def test_stat_by_index_multi(eng):
    index = [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
    ]
    data = fromlist([arange(12)], index=array(index).T, engine=eng)
    result = data.stat_by_index('sum', level=[0, 1])
    assert allclose(result.toarray(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    result = data.sum_by_index(level=[0, 1])
    assert allclose(result.toarray(), array([1, 14, 13, 38]))
    assert allclose(result.index, array([[0, 0], [0, 1], [1, 0], [1, 1]]))


def test_mean_by_panel(eng):
    data = fromlist([arange(8)], engine=eng)
    test1 = data.mean_by_panel(4)
    assert allclose(test1.index, array([0, 1, 2, 3]))
    assert allclose(test1.toarray(), [[2, 3, 4, 5]])
    test2 = data.mean_by_panel(2)
    assert allclose(test2.index, array([0, 1]))
    assert allclose(test2.toarray(), [[3, 4]])


def test_times_array(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2 = asarray([[7, 8], [9, 10], [11, 12]])
    mat1 = fromlist(mat1raw, engine=eng)
    truth = dot(mat1raw, mat2)
    result = mat1.times(mat2)
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(0, 2))


def test_times_array_alt(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2 = asarray([[7, 8, 7, 8], [9, 10, 9, 10], [11, 12, 11, 12]])
    mat1 = fromlist(mat1raw, engine=eng)
    truth = dot(mat1raw, mat2)
    result = mat1.times(mat2)
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(0, 4))


def test_times_vector(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2 = [7, 8, 9]
    mat1 = fromlist(mat1raw, engine=eng)
    truth = dot(mat1raw, mat2)
    result = mat1.times(mat2)
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, [0])


def test_times_scalar(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat2 = 5
    mat1 = fromlist(mat1raw, engine=eng)
    truth = mat1raw * mat2
    result = mat1.times(mat2)
    assert allclose(result.toarray(), truth)
    assert allclose(result.index, range(0, 3))


def test_gramian(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat1 = fromlist(mat1raw, engine=eng)
    result = mat1.gramian()
    truth = dot(mat1raw.T, mat1raw)
    assert allclose(result.toarray(), truth)


def test_cov(eng):
    mat1raw = asarray([[1, 2, 3], [4, 5, 6]])
    mat1 = fromlist(mat1raw, engine=eng)
    result = mat1.cov()
    truth = cov(mat1raw.T)
    assert allclose(result.toarray(), truth)
