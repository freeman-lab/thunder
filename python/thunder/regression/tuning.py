"""
Classes and standalone app for mass-unvariate tuning analyses
"""

import argparse
from scipy.io import loadmat
from numpy import array, sum, inner, dot, angle, abs, exp
from thunder.regression import RegressionModel
from thunder.utils import ThunderContext
from thunder.utils import save


class TuningModel(object):
    """Base class for loading and fitting tuning models

    Parameters
    ----------
    modelfile : str, or array
        Array of input values or location of a MAT file with name
        modelfile_s.mat containing a variable s with input values

    Attributes
    ----------
    s : array
        Input values along which tuning will be estimated,
        i.e. s if we are fitting a function y = f(s)
    """

    def __init__(self, modelfile):
        if type(modelfile) is str:
            self.s = loadmat(modelfile + "_s.mat")['s']
        else:
            self.s = modelfile

    @staticmethod
    def load(modelfile, tuningmode):
        return TUNING_MODELS[tuningmode](modelfile)

    def get(self, y):
        pass

    def fit(self, data):
        """Fit a mass univariate tuning model

        Parameters
        ----------
        data : RDD of (tuple, array) pairs
            The data to fit

        Returns
        -------
        params : RDD of (tuple, array) pairs
            Fitted tuning parameters for each record
        """
        return data.mapValues(lambda x: self.get(x))


class CircularTuningModel(TuningModel):
    """Class for circular tuning"""

    def get(self, y):
        """Estimate the circular mean and variance ("kappa"),
        identical to the max likelihood estimates of the
        parameters of the best fitting von-mises function
        """
        y = y - min(y)
        if sum(y) == 0:
            y += 1E-06
        y = y / sum(y)
        r = inner(y, exp(1j * self.s))
        mu = angle(r)
        v = abs(r) / sum(y)
        if v < 0.53:
            k = 2 * v + (v ** 3) + 5 * (v ** 5) / 6
        elif (v >= 0.53) & (v < 0.85):
            k = -.4 + 1.39 * v + 0.43 / (1 - v)
        elif (v ** 3 - 4 * (v ** 2) + 3 * v) == 0:
            k = array([0.0])
        else:
            k = 1 / (v ** 3 - 4 * (v ** 2) + 3 * v)
        if k > 1E8:
            k = array([0.0])
        return mu, k


class GaussianTuningModel(TuningModel):
    """Class for gaussian tuning"""

    def get(self, y):
        """Estimate the mean and variance,
        similar to the max likelihood estimates of the
        parameters of the best fitting gaussian
        but non-infinite supports may bias estimates
        """
        y[y < 0] = 0
        if sum(y) == 0:
            y += 1E-06
        y = y / sum(y)
        mu = dot(self.s, y)
        sigma = dot((self.s - mu) ** 2, y)
        return mu, sigma


TUNING_MODELS = {
    'circular': CircularTuningModel,
    'gaussian': GaussianTuningModel
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a parametric tuning curve to regression results")
    parser.add_argument("datafile", type=str)
    parser.add_argument("tuningmodelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("tuningmode", choices=("circular", "gaussian"), help="form of tuning curve")
    parser.add_argument("--regressmodelfile", type=str)
    parser.add_argument("--regressmode", choices=("linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()
    
    tsc = ThunderContext.start(appName="tuning")

    data = tsc.loadText(args.datafile, args.preprocess)
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
