from thunder.readers import LocalFileReader, LocalParallelReader


def make(tmpdir, files):
    tmpdir.mkdir('foo')
    tmpdir.mkdir('bar')
    tmpdir.mkdir('foo/bar')
    for f in files:
        tmpdir.join(f).write('hi')

def parse(files):
    return [f.split('/')[-1] for f in files]


def test_parallel_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_local_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_parallel_recursive_flat(tmpdir):
    filenames = ['b', 'a', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_local_recursive_flat(tmpdir):
    filenames = ['a', 'b', 'c']
    expected = ['a', 'b', 'c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_parallel_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['c']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_local_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['c']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=False)
    assert parse(actual) == expected


def test_parallel_recursive_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['a', 'c', 'b', 'q']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_local_recursive_nested(tmpdir):
    filenames = ['foo/b', 'foo/bar/q', 'bar/a', 'c']
    expected = ['a', 'c', 'b', 'q']
    make(tmpdir, filenames)
    actual = LocalFileReader().list(str(tmpdir), recursive=True)
    assert parse(actual) == expected


def test_tif_tiff_flat(tmpdir):
    filenames = ['b.tif', 'a.tif', 'c.tiff']
    expected = ['a.tif', 'b.tif', 'c.tiff']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), ext='tif', recursive=False)
    assert parse(actual) == expected
    actual = LocalParallelReader().list(str(tmpdir), ext='tif', recursive=True)
    assert parse(actual) == expected


def test_tif_tiff_recursive(tmpdir):
    filenames = ['foo/b.tif', 'foo/bar/q.tiff', 'bar/a', 'c.tif', 'd.tiff']
    expected = ['c.tif', 'd.tiff']
    make(tmpdir, filenames)
    actual = LocalParallelReader().list(str(tmpdir), ext='tif', recursive=False)
    assert parse(actual) == expected
    expected = ['c.tif', 'd.tiff', 'b.tif', 'q.tiff']
    actual = LocalParallelReader().list(str(tmpdir), ext='tif', recursive=True)
    assert parse(actual) == expected
