import pytest
import os
import glob
import json
from numpy import arange, allclose

from bolt import array as barray
from thunder.images.readers import fromlist, fromarray, frompng, fromtif, frombinary, fromexample

pytestmark = pytest.mark.usefixtures("eng")

resources = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


def test_from_list(eng):
    a = arange(8).reshape((2, 4))
    data = fromlist([a], engine=eng)
    assert allclose(data.shape, (1,) + a.shape)
    assert allclose(data.value_shape, a.shape)
    assert allclose(data.toarray(), a)


def test_from_array(eng):
    a = arange(8).reshape((1, 2, 4))
    data = fromarray(a, engine=eng)
    assert allclose(data.shape, a.shape)
    assert allclose(data.value_shape, a.shape[1:])
    assert allclose(data.toarray(), a)


def test_from_array_bolt(eng):
    a = arange(8).reshape((1, 2, 4))
    if eng is not None:
        b = barray(a, context=eng)
    else:
        b = barray(a)
    data = fromarray(b)
    assert allclose(data.shape, a.shape)
    assert allclose(data.value_shape, a.shape[1:])
    assert allclose(data.toarray(), a)


def test_from_array_single(eng):
    a = arange(8).reshape((2, 4))
    data = fromarray(a, engine=eng)
    assert allclose(data.shape, (1,) + a.shape)
    assert allclose(data.value_shape, a.shape)
    assert allclose(data.toarray(), a)


def test_from_png(eng):
    path = os.path.join(resources, 'singlelayer_png', 'dot1_grey.png')
    data = frompng(path, engine=eng)
    assert allclose(data.shape, (1, 70, 75))
    assert allclose(data.toarray().shape, (70, 75))
    assert allclose(data.toarray().max(), 239)
    assert allclose(data.toarray().min(), 1)


def test_from_png_keys(eng):
    path = os.path.join(resources, 'singlelayer_png', 'dot1_grey.png')
    data = frompng(path, engine=eng)
    assert (data.tordd().keys().first() == (0,) if data.mode == 'spark' else True)


def test_from_tif(eng):
    path = os.path.join(resources, "singlelayer_tif", "dot1_grey_lzw.tif")
    data = fromtif(path, engine=eng)
    assert allclose(data.shape, (1, 70, 75))
    assert allclose(data.toarray().shape, (70, 75))
    assert allclose(data.toarray().max(), 239)
    assert allclose(data.toarray().min(), 1)


def test_from_tif_keys(eng):
    path = os.path.join(resources, "singlelayer_tif", "dot1_grey_lzw.tif")
    data = fromtif(path, engine=eng)
    assert (data.tordd().keys().first() == (0,) if data.mode == 'spark' else True)


def test_from_tif_many(eng):
    path = os.path.join(resources, "singlelayer_tif", "dot*_grey_lzw.tif")
    data = fromtif(path, engine=eng)
    assert allclose(data.shape, (3, 70, 75))
    assert allclose(data.toarray().shape, (3, 70, 75))
    assert [x.sum() for x in data.toarray()] == [1233881, 1212169, 1191300]


def test_from_tif_multi_lzw(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw.tif')
    data = fromtif(path, engine=eng)
    val = data.toarray()
    assert allclose(data.shape, (1, 3, 70, 75))
    assert allclose(data.toarray().shape, (3, 70, 75))
    assert [val[i].sum() for i in range(3)] == [1140006, 1119161, 1098917]


def test_from_tif_multi_float(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_float32.tif')
    data = fromtif(path, engine=eng)
    val = data.toarray()
    assert allclose(data.shape, (1, 3, 70, 75))
    assert allclose(data.toarray().shape, (3, 70, 75))
    assert [val[i].sum() for i in range(3)] == [1140006, 1119161, 1098917]


def test_from_tif_multi_planes(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw.tif')
    data = fromtif(path, nplanes=3, engine=eng)
    assert data.shape[0] == 1
    data = fromtif(path, nplanes=1, engine=eng)
    assert data.shape[0] == 3
    assert allclose(data.toarray().shape, (3, 70, 75))
    assert [x.sum() for x in data.toarray()] == [1140006, 1119161, 1098917]


def test_from_tif_multi_planes_discard_extra(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw.tif')
    data = fromtif(path, nplanes=2, engine=eng, discard_extra=True)
    assert data.shape[0] == 1
    assert data.shape[1] == 2
    with pytest.raises(BaseException) as error_msg:
        data = fromtif(path, nplanes=2, engine=eng, discard_extra=False)
    assert 'nplanes' in str(error_msg.value)


def test_from_tif_multi_planes_many(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdotdot_lzw*.tif')
    data = fromtif(path, nplanes=3, engine=eng)
    assert allclose(data.shape, (2, 3, 70, 75))
    assert allclose(data.toarray().shape, (2, 3, 70, 75))
    data = fromtif(path, nplanes=1, engine=eng)
    assert allclose(data.shape, (6, 70, 75))
    assert allclose(data.toarray().shape, (6, 70, 75))
    assert [x.sum() for x in data.toarray()] == [
        1140006, 1119161, 1098917, 1140006, 1119161, 1098917]


def test_from_tif_multi_planes_variable(eng):
    path = os.path.join(resources, 'multilayer_tif', 'dotdot*_lzw*.tif')
    data = fromtif(path, nplanes=1, engine=eng)
    assert allclose(data.shape, (8, 70, 75))
    assert allclose(data.toarray().shape, (8, 70, 75))
    assert allclose(data[:, :, :].toarray().shape, (8, 70, 75))
    assert [x.sum() for x in data.toarray()] == [
        1140006, 1119161, 1140006, 1119161, 1098917, 1140006, 1119161, 1098917]


def test_from_tif_signed(eng):
    path = os.path.join(resources, 'multilayer_tif', 'test_signed.tif')
    data = fromtif(path, nplanes=1, engine=eng)
    assert data.dtype == 'int16'
    assert allclose(data.toarray().shape, (2, 120, 120))
    assert [x.sum() for x in data.toarray()] == [1973201, 2254767]


def test_from_binary(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((2, 4))
    a.tofile(os.path.join(str(tmpdir), 'test.bin'))
    data = frombinary(str(tmpdir), shape=(2, 4), dtype='int16', engine=eng)
    assert data.dtype == 'int16'
    assert allclose(data.shape, (1, 2, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_keys(tmpdir, eng):
    a = arange(8, dtype='int16').reshape((2, 4))
    a.tofile(os.path.join(str(tmpdir), 'test.bin'))
    data = frombinary(str(tmpdir), shape=(2, 4), dtype='int16', engine=eng)
    assert (data.tordd().keys().first() == (0,) if data.mode == 'spark' else True)


def test_from_binary_many(tmpdir, eng):
    a = [arange(8, dtype='int16').reshape((2, 4)), arange(8, 16, dtype='int16').reshape((2, 4))]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    data = frombinary(str(tmpdir), shape=(2, 4), dtype='int16', engine=eng)
    assert data.dtype == 'int16'
    assert allclose(data.shape, (2, 2, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_conf(tmpdir, eng):
    a = [arange(8, dtype='int32').reshape((2, 4)), arange(8, 16, dtype='int32').reshape((2, 4))]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    with open(os.path.join(str(tmpdir), 'conf.json'), 'w') as f:
        json.dump({'dims': [2, 4], 'dtype': 'int32'}, f)
    data = frombinary(str(tmpdir), engine=eng)
    assert data.dtype == 'int32'
    assert allclose(data.shape, (2, 2, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_multi(tmpdir, eng):
    a = arange(24, dtype='int16').reshape((2, 3, 4))
    a.tofile(os.path.join(str(tmpdir), 'test.bin'))
    data = frombinary(str(tmpdir), shape=(2, 3, 4), dtype='int16', engine=eng)
    assert data.dtype == 'int16'
    assert allclose(data.shape, (1, 2, 3, 4))
    assert allclose(data.toarray(), a)


def test_from_binary_multi_planes_many(tmpdir, eng):
    a1 = arange(16, dtype='int16').reshape((4, 2, 2))
    a2 = arange(16, 32, dtype='int16').reshape((4, 2, 2))
    a = [a1, a2]
    a[0].tofile(os.path.join(str(tmpdir), 'test0.bin'))
    a[1].tofile(os.path.join(str(tmpdir), 'test1.bin'))
    data = frombinary(str(tmpdir), shape=(4, 2, 2), dtype='int16', nplanes=2, engine=eng)
    assert allclose(data.shape, (2, 4, 2, 2))
    assert allclose(data.toarray().shape, (2, 4, 2, 2))
    data = frombinary(str(tmpdir), shape=(4, 2, 2), dtype='int16', nplanes=1, engine=eng)
    assert allclose(data.shape, (4, 4, 2))
    assert allclose(data.toarray().shape, (4, 4, 2))


def test_to_binary(tmpdir, eng):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromlist(a, engine=eng).tobinary(os.path.join(str(tmpdir), 'binary'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/binary/image*')]
    assert sorted(files) == ['image-00000.bin', 'image-00001.bin']
    with open(str(tmpdir) + '/binary/conf.json', 'r') as f:
        conf = json.load(f)
        assert conf['shape'] == [4, 2]
        assert conf['dtype'] == 'int16'


def test_to_binary_roundtrip(tmpdir, eng):
    a = [arange(8).reshape((4, 2)), arange(8, 16).reshape((4, 2))]
    data = fromlist(a, engine=eng)
    data.tobinary(os.path.join(str(tmpdir), 'images'))
    loaded = frombinary(os.path.join(str(tmpdir), 'images'), engine=eng)
    assert allclose(data.toarray(), loaded.toarray())


def test_to_binary_roundtrip_3d(tmpdir, eng):
    a = [arange(24).reshape((2, 3, 4)), arange(24, 48).reshape((2, 3, 4))]
    data = fromlist(a, engine=eng)
    data.tobinary(os.path.join(str(tmpdir), 'images'))
    loaded = frombinary(os.path.join(str(tmpdir), 'images'), engine=eng)
    assert allclose(data.toarray(), loaded.toarray())


def test_to_png(tmpdir, eng):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromlist(a, engine=eng).topng(os.path.join(str(tmpdir), 'images'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/images/image*')]
    assert sorted(files) == ['image-00000.png', 'image-00001.png']


def test_to_png_roundtrip(tmpdir, eng):
    a = [arange(8, dtype='uint8').reshape((4, 2))]
    data = fromlist(a, engine=eng)
    data.topng(os.path.join(str(tmpdir), 'images'), prefix='image')
    loaded = frompng(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())


def test_to_tif(tmpdir, eng):
    a = [arange(8, dtype='int16').reshape((4, 2)), arange(8, 16, dtype='int16').reshape((4, 2))]
    fromlist(a, engine=eng).totif(os.path.join(str(tmpdir), 'images'), prefix='image')
    files = [os.path.basename(f) for f in glob.glob(str(tmpdir) + '/images/image*')]
    assert sorted(files) == ['image-00000.tif', 'image-00001.tif']


def test_to_tif_roundtrip(tmpdir, eng):
    a = [arange(8, dtype='uint8').reshape((4, 2))]
    data = fromlist(a, engine=eng)
    data.totif(os.path.join(str(tmpdir), 'images'), prefix='image')
    loaded = fromtif(os.path.join(str(tmpdir), 'images'))
    assert allclose(data.toarray(), loaded.toarray())


def test_from_example(eng):
    return
    data = fromexample('fish', engine=eng)
    assert allclose(data.shape, (20, 76, 87, 2))
    data = fromexample('mouse', engine=eng)
    assert allclose(data.shape, (20, 64, 64))