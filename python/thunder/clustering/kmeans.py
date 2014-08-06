"""
Classes and standalone app for KMeans clustering
"""

import os
import argparse
import glob
from numpy import sum, array, argmin, corrcoef
from scipy.spatial.distance import cdist
from matplotlib import pyplot
import colorsys
import mpld3
from mpld3 import plugins
from pyspark import SparkContext
from thunder.utils import load
from thunder.utils import save
from thunder.viz.plugins import HiddenAxes
from thunder.viz.plots import scatter
from thunder.viz.colorize import Colorize
from thunder.utils.load import isrdd


class KMeansModel(object):
    """Class for an estimated KMeans model

    Parameters
    ----------
    centers : array
        The cluster centers
    """
    def __init__(self, centers):

        self.centers = centers

        # get unique colors for plotting
        n = len(self.centers)
        hsv_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        self.colors = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)

    def predict(self, data):
        """Predict the cluster that all data points belong to, and the similarity

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, a list of arrays, or a single array
            The data to predict cluster assignments on

        Returns
        -------
        closest : RDD of (tuple, array) pairs, list of arrays, or a single array
            For each data point, gives an array with the closest center for each data point,
            and the correlation with that center
        """

        if isrdd(data):
            return data.mapValues(lambda x: KMeans.similarity(x, self.centers))
        elif type(data) is list:
            return map(lambda x: KMeans.similarity(x, self.centers), data)
        else:
            return KMeans.similarity(data, self.centers)

    def plot(self, data, notebook=False, show=True, savename=None):

        fig = pyplot.figure()
        ncenters = len(self.centers)

        colorizer = Colorize()
        colorizer.get = lambda x: self.colors[int(self.predict(x)[0])]

        # plot time series of each center
        # TODO move into a time series plotting function in viz.plots
        for i, center in enumerate(self.centers):
            ax = pyplot.subplot2grid((ncenters, 3), (i, 0))
            ax.plot(center, color=self.colors[i], linewidth=5)
            fig.add_axes(ax)

        # make a scatter plot of the data
        ax2 = pyplot.subplot2grid((ncenters, 3), (0, 1), rowspan=ncenters, colspan=2)
        ax2, h2 = scatter(data, colormap=colorizer, ax=ax2)
        fig.add_axes(ax2)

        plugins.connect(fig, HiddenAxes())

        if show and notebook is False:
            mpld3.show()

        if savename is not None:
            mpld3.save_html(fig, savename)

        elif show is False:
            return mpld3.fig_to_html(fig)


class KMeans(object):
    """Class for KMeans clustering

    Parameters
    ----------
    k : int
        Number of clusters to find

    maxiter : int, optional, default = 20
        Maximum number of iterations to use

    tol : float, optional, default = 0.001
        Change tolerance for stopping algorithm
    """
    def __init__(self, k, maxiter=20, tol=0.001):
        self.k = k
        self.maxiter = maxiter
        self.tol = tol

    @staticmethod
    def closestpoint(p, centers):
        """Return the index of the closest point in centers to p"""
        return argmin(cdist(centers, array([p])))

    @staticmethod
    def similarity(p, centers):
        ind = argmin(cdist(centers, array([p])))
        corr = corrcoef(centers[ind], p)[0,1]
        return array([ind, corr])

    def train(self, data):
        """Train the clustering model using the standard
        k-means algorithm

        Parameters
        ----------
        data :  RDD of (tuple, array) pairs
            The data to cluster

        Returns
        -------
        centers : KMeansModel
            The estimated cluster centers
        """

        centers = array(map(lambda (_, v): v, data.takeSample(False, self.k)))
        tempdist = 1.0
        iter = 0

        while (tempdist > self.tol) & (iter < self.maxiter):
            closest = data.map(lambda (_, v): v).map(lambda p: (KMeans.closestpoint(p, centers), (p, 1)))
            pointstats = closest.reduceByKey(lambda (x1, y1), (x2, y2): (x1 + x2, y1 + y2))
            newpoints = pointstats.map(lambda (x, (y, z)): (x, y / z)).collect()
            tempdist = sum(sum((centers[x] - y) ** 2) for (x, y) in newpoints)

            for (i, j) in newpoints:
                centers[i] = j

            iter += 1

        return KMeansModel(centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do kmeans clustering")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "kmeans")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    model = KMeans(k=args.k, maxiter=args.maxiter, tol=args.tol).train(data)
    labels = model.predict(data)

    outputdir = args.outputdir + "-kmeans"
    save(model.centers, outputdir, "centers", "matlab")
    save(labels, outputdir, "labels", "matlab")
