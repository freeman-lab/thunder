from numpy import amin, amax, shape, array, transpose, asarray, std
from thunder.utils.load import isrdd
from thunder.utils.save import pack
from thunder.viz import Colorize
from matplotlib import pyplot


def imagemap(data, colormap='polar', scale=1.0, ax=None):
    """Create an image from a collection of key-value pairs, using the
    keys as spatial indices, and the values to compute colors"""

    if ax is None:
        ax = pyplot.gca()

    if isrdd(data):
        ndim = len(data.first()[0])
        data = Colorize(colormap, scale).calc(data)
        if ndim == 3:
            pixels = pack(data, axes=2)
        elif ndim == 2:
            pixels = pack(data)
        else:
            raise Exception('number of spatial dimensions for images must be 2 or 3')
    else:
        raise Exception('input must be an RDD')

    h = ax.imshow(transpose(pixels, [2, 1, 0]))
    return ax, h


def pointmap(data, colormap='polar', scale=1.0, ax=None):
    """Create a spatial point map from a collection of key-value pairs, using the
    keys as spatial indices, and the values to compute colors"""

    if ax is None:
        ax = pyplot.gca()

    if isrdd(data):
        pts = Colorize(colormap, scale).calc(data).collect()
    else:
        raise Exception('input must be an RDD')

    clrs = array(map(lambda (k, v): v, pts))
    x = map(lambda (k, v): k[0], pts)
    y = map(lambda (k, v): k[1], pts)
    z = map(lambda (k, v): k[2], pts)  # currently unused
    h = ax.scatter(x, y, s=100, c=clrs, alpha=0.5, edgecolor='black', linewidth=0.2)
    return ax, h


def scatter(pts, nsamples=100, colormap=None, scale=1, thresh=0.001, ax=None, store=False):
    """Create a scatter plot of x and y points from an array or an RDD (through sampling)
    Can optionally use the values to determine colors"""

    if ax is None:
        ax = pyplot.gca()

    if isrdd(pts):
        if thresh is not None:
            pts = array(pts.values().filter(lambda x: std(x) > thresh).takeSample(False, nsamples))
        else:
            pts = array(pts.values().takeSample(False, nsamples))
        if len(pts) == 0:
            raise Exception('no samples found, most likely your threshold is too low')
    else:
        pts = asarray(pts)

    if colormap is not None:
        # pass in strings or actual colormap objects
        if isinstance(colormap, basestring):
            clrs = Colorize(colormap, scale).calc(pts)
        else:
            clrs = colormap.calc(pts)

    else:
        clrs = 'indianred'

    h = ax.scatter(pts[:, 0], pts[:, 1], s=100, c=clrs, alpha=0.6, edgecolor='black', linewidth=0.2)

    if store is True:
        return ax, h, pts
    else:
        return ax, h


def tsrecon(tsbase, samples, ax=None):
    """Plot a single time series to represent a reconstruction using some low dimensional basis,
    and use a set of weights to create many possible lines for interactive plotting"""

    if ax is None:
        ax = pyplot.gca()

    t = shape(tsbase)[1]
    tt = range(0, t)

    h = ax.plot(array(tt), 0 * array(tt), '-w', lw=5, alpha=0.5)
    ax.set_ylim(amin(tsbase)/4, amax(tsbase)/4)

    recon = map(lambda x: (x[0] * tsbase[0, :] + x[1] * tsbase[1, :]).tolist(), samples)
    topairs = lambda y: map(lambda x: list(x), zip(tt, y))
    data = map(topairs, recon)

    return ax, h, data