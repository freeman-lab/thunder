"""
Standalone app for mass-unvariate regression
"""

import os
import argparse
import glob
from thunder.regression import RegressionModel
from thunder.utils import load
from thunder.utils import save
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "regress")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])
    
    data = load(sc, args.datafile, args.preprocess)
    stats, betas, resid = RegressionModel.load(args.modelfile, args.regressmode).fit(data)

    outputdir = args.outputdir + "-regress"
    save(stats, outputdir, "stats", "matlab")
    save(betas, outputdir, "betas", "matlab")
