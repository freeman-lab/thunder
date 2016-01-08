"""
Class for Principal Component Analysis
"""

from thunder.factorization.svd import SVD
from thunder.rdds.series import Series
from thunder.rdds.matrices import RowMatrix


class PCA(object):
    """
    Principal components analysis on a distributed matrix.

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdMethod : str, optional, default = "auto"
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k principal components

    `latent` : array, shape (k,)
        The latent values

    `scores` : RowMatrix, nrows, each of shape (k,)
        The scores (i.e. the representation of the data in PC space)

    See also
    --------
    SVD : singular value decomposition
    """

    def __init__(self, k=3, svdMethod='auto'):
        self.k = k
        self.svdMethod = svdMethod
        self.scores = None
        self.latent = None
        self.comps = None

    def fit(self, data):
        """
        Estimate principal components

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()

        mat = data.center(1)

        svd = SVD(k=self.k, method=self.svdMethod)
        svd.calc(mat)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        # taken from/modelled after sklearn, see:
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/pca.py#L274-L277
        self.explained_variance_ = (self.svd.s ** 2) / data.count()
        self.explained_variance_ratio_ = (self.explained_variance_ / self.explained_variance_.sum())
        
        return self

    def transform(self, data):
        """
        Project data into principal component space

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        -------
        scores : RowMatrix, nrows, each of shape (k,)
            The scores (i.e. the representation of the data in PC space)
        """

        if not (isinstance(data, Series)):
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        mat = data.center(1)
        scores = mat.times(self.comps.T / self.latent)
        return scores
