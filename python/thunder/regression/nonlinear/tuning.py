"""
Classes for mass-unvariate tuning analyses
"""

from numpy import array, sum, inner, dot, angle, abs, exp, asarray

from thunder.rdds.series import Series
from thunder.utils.common import loadMatVar



class TuningModel(object):
    """
    Base class for loading and fitting tuning models.

    Parameters
    ----------
    modelFile : str, or array
        Array of input values or specification of a MAT file
        containing a variable s with input values

    var : str, default = 's'
        Variable name if loading from a MAT file

    Attributes
    ----------
    s : array
        Input values along which tuning will be estimated,
        i.e. s if we are fitting a function y = f(s)

    See also
    --------
    CircularTuningModel : circular tuning parameter estimation
    GaussianTuningModel : gaussian tuning parameter estimation
    """

    def __init__(self, modelFile, var='s'):
        if isinstance(modelFile, basestring):
            self.s = loadMatVar(modelFile, var)
        else:
            self.s = modelFile

    @staticmethod
    def load(modelFile, tuningMode):
        from thunder.utils.common import checkParams
        checkParams(tuningMode.lower(), TUNING_MODELS.keys())
        return TUNING_MODELS[tuningMode.lower()](modelFile)

    def get(self, y):
        pass

    def fit(self, data):
        """
        Fit a mass univariate tuning model.

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            The data to fit tuning models to, a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        -------
        params : RDD of (tuple, array) pairs
            Fitted tuning parameters for each record
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        return Series(data.rdd.mapValues(lambda x: self.get(x)), index=['center', 'spread']).__finalize__(data)


class CircularTuningModel(TuningModel):
    """ Circular tuning model fitting. """

    def get(self, y):
        """
        Estimate the circular mean and variance ("kappa"),
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
        return asarray([mu, k])


class GaussianTuningModel(TuningModel):
    """ Gaussian tuning model fitting. """

    def get(self, y):
        """
        Estimate the mean and variance,
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
        return asarray([mu, sigma])


TUNING_MODELS = {
    'circular': CircularTuningModel,
    'gaussian': GaussianTuningModel
}
