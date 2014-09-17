"""
Standalone app for mass-unvariate regression combined with PCA
"""

import argparse
from thunder.regression import RegressionModel
from thunder.factorization import PCA
from thunder.utils import ThunderContext
from thunder.utils import save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()
    
    tsc = ThunderContext.start(appName="regresswithpca")

    data = tsc.loadText(args.datafile, args.preprocess)
    model = RegressionModel.load(args.modelfile, args.regressmode)  # do regression
    betas, stats, resid = model.fit(data)
    pca = PCA(args.k).fit(betas)  # do PCA
    traj = model.fit(data, pca.comps)  # get trajectories

    outputdir = args.outputdir + "-regress"
    save(pca.comps, outputdir, "comps", "matlab")
    save(pca.latent, outputdir, "latent", "matlab")
    save(pca.scores, outputdir, "scores", "matlab")
    save(traj, outputdir, "traj", "matlab")