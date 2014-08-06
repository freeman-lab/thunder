"""
Standalone app for mass-unvariate tuning analyses
"""

import os
import argparse
import glob
from thunder.regression import RegressionModel, TuningModel
from thunder.utils import load
from thunder.utils import save
from pyspark import SparkContext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("tuningmodelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("tuningmode", choices=("circular", "gaussian"), help="form of tuning curve")
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)
    parser.add_argument("--regressmodelfile", type=str)
    parser.add_argument("--regressmode", choices=("linear", "bilinear"), help="form of regression")

    args = parser.parse_args()
    
    sc = SparkContext(args.master, "tuning")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess)
    tuningmodel = TuningModel.load(args.tuningmodelfile, args.tuningmode)
    if args.regressmodelfile is not None:
        # use regression results
        regressmodel = RegressionModel.load(args.regressmodelfile, args.regressmode)
        betas, stats, resid = regressmodel.fit(data)
        params = tuningmodel.fit(betas)
    else:
        # use data
        params = tuningmodel.fit(data)

    outputdir = args.outputdir + "-tuning"
    save(params, outputdir, "params", "matlab")
