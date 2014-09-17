"""
Class and standalone app for calculating time series statistics
"""

import argparse
from numpy import median, mean, std
from scipy.linalg import norm
from thunder.timeseries.base import TimeSeriesBase
from thunder.utils import ThunderContext, save


class Stats(TimeSeriesBase):
    """Class for computing simple summary statistics"""

    def __init__(self, statistic):
        self.func = {
            'median': lambda x: median(x),
            'mean': lambda x: mean(x),
            'std': lambda x: std(x),
            'norm': lambda x: norm(x - mean(x)),
        }[statistic]

    def get(self, y):
        """Compute the statistic"""

        return self.func(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std", "norm"), help="which summary statistic")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="stats")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    vals = Stats(args.mode).calc(data)

    outputdir = args.outputdir + "-stats"
    save(vals, outputdir, "stats_" + args.mode, "matlab")