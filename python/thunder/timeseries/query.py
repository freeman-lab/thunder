"""
Class and standalone app for querying
"""

import argparse
from numpy import zeros, mean
from scipy.io import loadmat
from thunder.utils import ThunderContext, getdims, subtoind, indtosub
from thunder.utils import save


class Query(object):
    """Class for querying key-value data by averaging together
    records with the specified indices (typically, the keys
    are spatial coordinates and the values are time series,
    so this returns the center coordinates and average time series
    of the specified records).

    Parameters
    ----------
    indsfile : str, or array-like (2D)
        Array of indices, each an array-like of integer indices, or
        filename of a MAT file containing a set of indices as a cell array
        stored in the variable inds

    Attributes
    ----------

    `inds` : list
        The loaded indices (see parameters)

    `n` : int
        Number of indices

    `keys` : array, shape (n, k) where k is the length of each value
        Averaged values

    `values` : array, shape (n, d) where d is the number of keys
        Averaged keys


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
        self : returns an instance of self.
        """

        dims = getdims(data)
        data = subtoind(data, dims.max)

        # loop over indices, computing average keys and average values
        keys = zeros((self.n, len(dims.count())))
        values = zeros((self.n, len(data.first()[1])))
        for idx, indlist in enumerate(self.inds):
            if len(indlist) > 0:
                values[idx, :] = self.select(data, idx).map(lambda (k, x): x).sum() / len(indlist)
                keys[idx, :] = mean(map(lambda (k, v): k, indtosub(map(lambda k: (k, 0), indlist), dims.max)), axis=0)

        self.keys = keys
        self.values = values

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="query data by averaging values for given indices")
    parser.add_argument("datafile", type=str)
    parser.add_argument("indsfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="query")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    qry = Query(args.indsfile).calc(data)

    outputdir = args.outputdir + "-query"
    save(qry.keys, outputdir, "centers", "matlab")
    save(qry.values, outputdir, "ts", "matlab")
