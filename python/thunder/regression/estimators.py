from numpy import mean, insert, dot, hstack, zeros, array
from scipy.linalg import inv

class RegressionEstimator(object):
    """
    Abstract base class for all regression fitting procedures
    """

    def __init__(self, X, intercept=False):
        self.X = X
        self.intercept = intercept

    def estimate(self, y):
        raise NotImplementedError

    def fit(self, y):
        if self.intercept:
            b0 = mean(y)
            y = y - b0

        b = self.estimate(y)

        if self.intercept:
            b = insert(b, 0, b0)
        return b

class PseudoInv(RegressionEstimator):
    """
    Class for fitting regression models via a psuedo-inverse
    """

    def __init__(self, X, **kwargs):
        super(PseudoInv, self).__init__(X, **kwargs)
        self.Xhat = dot(inv(dot(X.T, X)), X.T)

    def estimate(self, y):
        return dot(self.Xhat, y)

class TikhonovPseudoInv(PseudoInv):
    """
    Class for fitting Tikhonov regularization models via a psuedo-inverse
    """

    def __init__(self, X, nPenalties, **kwargs):
        self.nPenalties = nPenalties
        super(TikhonovPseudoInv, self).__init__(X, **kwargs)

    def estimate(self, y):
        y = hstack([y, zeros(self.nPenalties)])
        return super(TikhonovPseudoInv, self).estimate(y)

