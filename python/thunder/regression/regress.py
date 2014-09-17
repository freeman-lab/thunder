"""
Classes and standalone app for mass-unvariate regression
"""

import argparse
from scipy.io import loadmat
from numpy import sum, outer, inner, mean, shape, dot, transpose, concatenate, ones
from scipy.linalg import inv
from thunder.utils import ThunderContext
from thunder.utils import save


class RegressionModel(object):
    """Base class for loading and fitting regression models"""

    @staticmethod
    def load(modelfile, regressmode, **opts):
        return REGRESSION_MODELS[regressmode](modelfile, **opts)

    #TODO add a make method to construct regressors from parameters

    def get(self, y):
        pass

    def fit(self, data, comps=None):
        """Fit mass univariate regression models

        Parameters
        ----------
        data : RDD of (tuple, array) pairs
            The data to fit

        Returns
        -------
        betas : RDD of (tuple, array) pairs
            Fitted model parameters for each record

        stats : RDD of (tuple, float) pairs
            Model fit statistic for each record

        resid : RDD of (tuple, array) pairs
            Residuals for each record
        """

        if comps is not None:
            traj = data.map(lambda (_, v): v).map(
                lambda x: outer(x, inner(self.get(x)[0] - mean(self.get(x)[0]), comps))).reduce(
                    lambda x, y: x + y) / data.count()
            return traj
        else:
            result = data.mapValues(lambda x: self.get(x))
            betas = result.mapValues(lambda x: x[0])
            stats = result.mapValues(lambda x: x[1])
            resid = result.mapValues(lambda x: x[2])
            return betas, stats, resid


class MeanRegressionModel(RegressionModel):
    """Class for regression in the form of simple averaging
    via multiplication by a design matrix

    Parameters
    ----------
    modelfile : array, or string
        Array contaiing design matrix, or location of a MAT file
        with name modelfile_X.mat containing a variable X

    Attributes
    ----------
    x : array
        The design matrix

    xhat : array
        Pseudoinverse of the design matrix
    """

    def __init__(self, modelfile):
        if type(modelfile) is str:
            x = loadmat(modelfile + "_X.mat")['X']
        else:
            x = modelfile
        x = x.astype(float)
        x_hat = (x.T / sum(x, axis=1)).T
        self.x = x
        self.x_hat = x_hat

    def get(self, y):
        """Compute regression coefficients, r2 statistic, and residuals"""

        b = dot(self.x_hat, y)
        predic = dot(b, self.x)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst
        return b, r2, resid


class LinearRegressionModel(RegressionModel):
    """Class for ordinary least squares linear regression

    Parameters
    ----------
    modelfile : array, or string
        Array contaiing design matrix, or location of a MAT file
        with name modelfile_X.mat containing a variable X

    Attributes
    ----------
    x : array
        The design matrix

    xhat : array
        Pseudoinverse of the design matrix
    """

    def __init__(self, modelfile):
        if type(modelfile) is str:
            x = loadmat(modelfile + "_X.mat")['X']
        else:
            x = modelfile
        x = concatenate((ones((1, shape(x)[1])), x))
        x_hat = dot(inv(dot(x, transpose(x))), x)
        self.x = x
        self.x_hat = x_hat

    def get(self, y):
        """Compute regression coefficients, r2 statistic, and residuals"""

        b = dot(self.x_hat, y)
        predic = dot(b, self.x)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst
        return b[1:], r2, resid


class BilinearRegressionModel(RegressionModel):
    """Class for bilinear regression with two design matrices

    Parameters
    ----------
    modelfile : tuple(array), or string
        Tuple of arrays each contaiing a design matrix,
        or location of MAT files with names modelfile_X1.mat and
        modelfile_X2.mat containing variables X1 and X2

    Attributes
    ----------
    x1 : array
        The first design matrix

    x2 : array
        The second design matrix

    x1hat : array
        Pseudoinverse of the first design matrix
    """

    def __init__(self, modelfile):
        if type(modelfile) is str:
            x1 = loadmat(modelfile + "_X1.mat")['X1']
            x2 = loadmat(modelfile + "_X2.mat")['X2']
        else:
            x1 = modelfile[0]
            x2 = modelfile[1]
        x1_hat = dot(inv(dot(x1, transpose(x1))), x1)
        self.x1 = x1
        self.x2 = x2
        self.x1_hat = x1_hat

    def get(self, y):
        """Compute regression coefficients from the second design matrix,
        a single r2 statistic, and residuals for the full model"""

        b1 = dot(self.x1_hat, y)
        b1 = b1 - min(b1)
        b1_hat = dot(transpose(self.x1), b1)
        if sum(b1_hat) == 0:
            b1_hat += 1E-06
        x3 = self.x2 * b1_hat
        x3 = concatenate((ones((1, shape(x3)[1])), x3))
        x3_hat = dot(inv(dot(x3, transpose(x3))), x3)
        b2 = dot(x3_hat, y)
        predic = dot(b2, x3)
        resid = y - predic
        sse = sum((predic - y) ** 2)
        sst = sum((y - mean(y)) ** 2)
        if sst == 0:
            r2 = 0
        else:
            r2 = 1 - sse / sst

        return b2[1:], r2, resid

REGRESSION_MODELS = {
    'linear': LinearRegressionModel,
    'bilinear': BilinearRegressionModel,
    'mean': MeanRegressionModel
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("modelfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("regressmode", choices=("mean", "linear", "bilinear"), help="form of regression")
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="regress")

    data = tsc.loadText(args.datafile, args.preprocess)
    stats, betas, resid = RegressionModel.load(args.modelfile, args.regressmode).fit(data)

    outputdir = args.outputdir + "-regress"
    save(stats, outputdir, "stats", "matlab")
    save(betas, outputdir, "betas", "matlab")
