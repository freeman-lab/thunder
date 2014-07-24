"""
Class and standalone app for Principal Component Analysis
"""

import os
import argparse
import glob
from matplotlib import pyplot
import mpld3
from mpld3 import plugins
from pyspark import SparkContext
from thunder.factorization import SVD
from thunder.utils import load
from thunder.utils import save
from thunder.utils.matrices import RowMatrix
from thunder.viz.plugins import LinkedView, HiddenAxes
from thunder.viz.plots import pointmap, imagemap, scatter, tsrecon


class PCA(object):
    """Perform principal components analysis
    using the singular value decomposition

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdmethod : str, optional, default = "direct"
        Which method to use for performing the SVD

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k principal components

    `latent` : array, shape (k,)
        The latent values

    `scores` : RDD of nrows (tuple, array) pairs, each of shape (k,)
        The scores (i.e. the representation of the data in PC space)
    """

    def __init__(self, k=3, svdmethod='direct'):
        self.k = k
        self.svdmethod = svdmethod

    def fit(self, data):
        """Estimate principal components

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, or RowMatrix
        """

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        data.center(0)
        svd = SVD(k=self.k, method=self.svdmethod)
        svd.calc(data)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        return self

    def plot(self, notebook=False, colormap='polar', scale=1, maptype='points', show=True, savename=None):

        # make a spatial map based on the scores
        fig = pyplot.figure(figsize=(12, 5))
        ax1 = pyplot.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
        if maptype is 'points':
            ax1, h1 = pointmap(self.scores, colormap=colormap, scale=scale, ax=ax1)
        elif maptype is 'image':
            ax1, h1 = imagemap(self.scores, colormap=colormap, scale=scale, ax=ax1)
        fig.add_axes(ax1)

        # make a scatter plot of sampled scores
        ax2 = pyplot.subplot2grid((2, 3), (1, 0))
        ax2, h2, samples = scatter(self.scores, colormap=colormap, scale=scale, thresh=0.01, nsamples=1000, ax=ax2, store=True)
        fig.add_axes(ax2)

        # make the line plot of reconstructions from principal components for the same samples
        ax3 = pyplot.subplot2grid((2, 3), (0, 0))
        ax3, h3, linedata = tsrecon(self.comps, samples, ax=ax3)

        plugins.connect(fig, LinkedView(h2, h3[0], linedata))
        plugins.connect(fig, HiddenAxes())

        if show and notebook is False:
            mpld3.show()

        if savename is not None:
            mpld3.save_html(fig, savename)

        elif show is False:
            return mpld3.fig_to_html(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "pca")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    result = PCA(args.k, args.svdmethod).fit(data)

    outputdir = args.outputdir + "-pca"
    save(result.comps, outputdir, "comps", "matlab")
    save(result.latent, outputdir, "latent", "matlab")
    save(result.scores, outputdir, "scores", "matlab")