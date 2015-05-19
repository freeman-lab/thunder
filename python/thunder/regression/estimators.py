class RegressionEstimator(object):
    '''
    Abstract base class for all regression fitting procedures
    '''

    def __init__(self, intercept=False):
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
    '''
    Class for fitting regression models via a psuedo-inverse
    '''

    def __init__(self, X, **kwargs):
        super(PseudoInv, self).__init__(**kwargs)
        self.Xhat = dot(inv(dot(X.T, X)), X.T)

    def estimate(self, y):
        return dot(self.Xhat, y)

class TikhonovPseudoInv(PseudoInv):
    '''
    Class for fitting Tikhonov regularization models via a psuedo-inverse
    '''

    def __init__(self, X, nPenalties, **kwargs):
        self.nPenalties = nPenalties
        super(TikhonovPseudoInv, self).__init__(X, **kwargs)

    def estimate(self, y):
        y = hstack([y, zeros(self.nPenalties)])
        return super(TikhonovPseudoInv, self).estimate(y)


class QuadProg(RegressionEstimator):
    '''
    Class for fitting regression models via quadratic programming

    cvxopt.solvers.qp minimizes (1/2)*x'*P*x + q'*x with the constraint Ax <= b
    '''

    def __init__(self, X, A, b, **kwargs):
        super(QuadProg, self).__init__(**kwargs)
        self.X = X
        self.P = cvxoptMatrix(dot(X.T, X))
        self.A = cvxoptMatrix(A)
        self.b = cvxoptMatrix(b)

    def estimate(self, y):
        from cvxopt.solvers import qp, options
        options['show_progress'] = False
        q = cvxoptMatrix(array(dot(-self.X.T, y), ndmin=2).T)
        return array(qp(self.P, q, self.A, self.b)['x']).flatten()

