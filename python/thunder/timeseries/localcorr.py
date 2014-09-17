"""
Class and standalone app for local correlation
"""

import argparse
from numpy import corrcoef
from thunder.utils import ThunderContext, save, indtosub, subtoind, getdims


class LocalCorr(object):
    """Class for computing local correlations

    Parameters
    ----------
    neighborhood : int, optional, default = 3
        Size of spatial neighborhood
    """
    def __init__(self, neighborhood=3):
        self.neighborhood = neighborhood

    def calc(self, data):
        """Compute correlation between every data point
        and the average of a local neighborhood,
        by correlating each data point with the average of a
        local neighborhood in x and y (typically time series data)

        Parameters
        ----------
        data : RDD of (tuple, array) pairs
            The data to compute correlations on

        Returns
        -------
        corr : RDD of (tuple, float) pairs
            The local correlation for each record, sorted by keys
        """

        def clip(val, mn, mx):
            """Clip a value below by mn and above by mx"""
            if val < mn:
                return mn
            if val > mx:
                return mx
            else:
                return val

        def maptoneighborhood(ind, ts, sz, mn, mx):
            """Create a list of key value pairs with multiple shifted copies
            of the time series ts over a region specified by sz
            """
            rng_x = range(-sz, sz+1, 1)
            rng_y = range(-sz, sz+1, 1)
            out = list()
            for x in rng_x:
                for y in rng_y:
                    new_x = clip(ind[0] + x, mn[0], mx[0])
                    new_y = clip(ind[1] + y, mn[1], mx[1])
                    newind = (new_x, new_y, ind[2])
                    out.append((newind, ts))
            return out

        # get boundaries using dimension keys
        dims = getdims(data)

        if len(dims.max) not in [2, 3]:
            raise NotImplementedError('keys must have 2 or 3 dimensions to compute local correlations')

        # flat map to key value pairs where the key is neighborhood identifier and value is time series
        neighbors = data.flatMap(lambda (k, v): maptoneighborhood(k, v, self.neighborhood, dims.min[0:2], dims.max[0:2]))

        # reduce by key to get the average time series for each neighborhood
        means = neighbors.reduceByKey(lambda x, y: x + y).mapValues(lambda x: x / ((2*self.neighborhood+1)**2))

        # join with the original time series data to compute correlations
        result = data.join(means)

        # get correlations
        corr = result.mapValues(lambda x: corrcoef(x[0], x[1])[0, 1])

        # force sorting, but reverse keys for correct ordering
        return corr.map(lambda (k, v): (k[::-1], v)).sortByKey().map(lambda (k, v): (k[::-1], v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="correlate time series with neighbors")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("sz", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="localcorr")

    data = tsc.loadText(args.datafile, filter=args.preprocess).cache()
    corrs = LocalCorr(args.sz).calc(data)

    outputdir = args.outputdir + "-localcorr"
    save(corrs, outputdir, "corr", "matlab")