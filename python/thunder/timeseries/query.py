"""
Class and standalone app for querying
"""

import os
import argparse
import glob
from numpy import zeros
from scipy.io import loadmat
from pyspark import SparkContext
from thunder.utils import load, getdims, subtoind
from thunder.utils import save


class Query(object):
    """Class for querying time series data by averaging together
    data points with the specified indices

    Parameters
    ----------
    indsfile : str, or array-like (2D)
        Array of indices, each an array-like of integer indices, or
        filename of a MAT file containing a set of indices as a cell array
        stored in the variable inds
    """

    def __init__(self, indsfile):
        if type(indsfile) is str:
            inds = loadmat(indsfile)['inds'][0]
        else:
            inds = indsfile
        self.inds = inds
        self.n = len(inds)

    def select(self, data, i):
        """Subselect rows of data with a given set of integer indices

        Parameters
        ----------
        data : RDD of (int, array) pairs, each array of shape (ncols,)
            Data to subselect rows from

        i : int
            Which set of indices to use

        Returns
        -------
        subset : RDD of (int, array) pairs
            Data with subset of rows
        """

        inds_set = set(self.inds[i].flat)
        inds_bc = data.context.broadcast(inds_set)
        subset = data.filter(lambda (k, _): k in inds_bc.value)
        return subset

    def calc(self, data):
        """Calculate averages. Keys (tuples) are converted
        into linear indices based on their dimensions

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, each array of shape (ncols,)
            Data to compute averages from

        Returns
        -------
        ts : array, shape (n, ncols)
        """

        dims = getdims(data)
        data = subtoind(data, dims.max)

        # loop over indices, averaging time series
        ts = zeros((self.n, len(data.first()[1])))
        for i in range(0, self.n):
            ts[i, :] = self.select(data, i).map(lambda (k, x): x).sum() / len(self.inds[i])

        return ts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query time series data by averaging values for given indices")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("indsfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-percentile", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "query")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    ts = Query(args.indsfile).calc(data)

    outputdir = args.outputdir + "-query"
    save(ts, outputdir, "ts", "matlab")
