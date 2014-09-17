"""
Class and standalone app for Principal Component Analysis
"""

import argparse
from thunder.factorization import SVD
from thunder.utils import ThunderContext, RowMatrix, save


class PCA(object):
    """Perform principal components analysis
    using the singular value decomposition

    Parameters
    ----------
    k : int
        Number of principal components to estimate

    svdmethod : str, optional, default = "direct"
        Which method to use for performing the SVD

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k principal components

    `latent` : array, shape (k,)
        The latent values

    `scores` : RDD of nrows (tuple, array) pairs, each of shape (k,)
        The scores (i.e. the representation of the data in PC space)
    """

    def __init__(self, k=3, svdmethod='direct'):
        self.k = k
        self.svdmethod = svdmethod

    def fit(self, data):
        """Estimate principal components

        Parameters
        ----------
        data : RDD of (tuple, array) pairs, or RowMatrix
        """

        if type(data) is not RowMatrix:
            data = RowMatrix(data)

        data.center(0)
        svd = SVD(k=self.k, method=self.svdmethod)
        svd.calc(data)

        self.scores = svd.u
        self.latent = svd.s
        self.comps = svd.v

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do principal components analysis")
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--svdmethod", choices=("direct", "em"), default="direct", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start(appName="pca")

    data = tsc.loadText(args.datafile, args.preprocess).cache()
    result = PCA(args.k, args.svdmethod).fit(data)

    outputdir = args.outputdir + "-pca"
    save(result.comps, outputdir, "comps", "matlab")
    save(result.latent, outputdir, "latent", "matlab")
    save(result.scores, outputdir, "scores", "matlab")