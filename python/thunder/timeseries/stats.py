"""
Standalone app for calculating time series statistics
"""

import os
import argparse
import glob
from pyspark import SparkContext
from thunder.timeseries import Stats
from thunder.utils import load
from thunder.utils import save



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute summary statistics on time series data")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("mode", choices=("mean", "median", "std", "norm"), help="which summary statistic")
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(appName="stats")

    data = load(sc, args.datafile, args.preprocess).cache()
    vals = Stats(args.mode).calc(data)

    outputdir = args.outputdir + "-stats"
    save(vals, outputdir, "stats_" + args.mode, "matlab")