"""
Class and standalone app for cross correlations
"""

import argparse
from numpy import mean, zeros, roll, shape, dot
from scipy.linalg import norm
from scipy.io import loadmat
from thunder.timeseries.base import TimeSeriesBase
from thunder.factorization import PCA
from thunder.utils import ThunderContext, save


class CrossCorr(TimeSeriesBase):
    """Class for computing lagged cross correlations

    Parameters
    ----------
    x : str, or array
        Signal to cross-correlate with, can be an array
        or location of MAT file with name sigfile_X.mat
        containing variable X with signal

    Attributes
    ----------
    x : array
        Signal to cross-correlate with
    """

    def __init__(self, sigfile, lag):
        if type(sigfile) is str:
            x = loadmat(sigfile + "_X.mat")['X'][0]
        else:
            x = sigfile
        x = x - mean(x)
        x = x / norm(x)

        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(x)
            m = len(shifts)
            x_shifted = zeros((m, d))
            for ix in range(0, len(shifts)):
                tmp = roll(x, shifts[ix])
                if shifts[ix] < 0:  # zero padding
                    tmp[(d+shifts[ix]):] = 0
                if shifts[ix] > 0:
                    tmp[:shifts[ix]] = 0
                x_shifted[ix, :] = tmp
            self.x = x_shifted
        else:
            self.x = x

    def get(self, y):
        """Compute cross correlation between y and x"""

        y = y - mean(y)
        n = norm(y)
        if n == 0:
            b = zeros((shape(self.x)[0],))
        else:
            y /= norm(y)
            b = dot(self.x, y)
        return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("sigfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("lag", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="crosscorr")

    data = tsc.loadText(args.datafile, args.preprocess).cache()

    outputdir = args.outputdir + "-crosscorr"

    # post-process data with pca if lag greater than 0
    vals = CrossCorr(args.sigfile, args.lag).calc(data)
    if args.lag is not 0:
        out = PCA(2).fit(vals)
        save(out.comps, outputdir, "comps", "matlab")
        save(out.latent, outputdir, "latent", "matlab")
        save(out.scores, outputdir, "scores", "matlab")
    else:
        save(vals, outputdir, "betas", "matlab")
