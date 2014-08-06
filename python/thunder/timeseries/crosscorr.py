"""
Standalone app for cross correlations
"""

import os
import argparse
import glob
from pyspark import SparkContext
from thunder.timeseries import CrossCorr
from thunder.factorization import PCA
from thunder.utils import load
from thunder.utils import save



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("sigfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("lag", type=int)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "crosscorr")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])
    
    data = load(sc, args.datafile, args.preprocess).cache()

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
