"""
Class for performing non-negative matrix factorization
"""

import argparse, glob, os
import numpy as np
from pyspark import SparkContext
from thunder.utils import load, save


class NMF(object):
    """
    Large-scale non-negative matrix factorization on a dense matrix
    represented as an RDD with nrows and ncols

    Parameters
    ----------
    method : string, optional, default 'als'
        Specifies which iterative algorithm is to be used. Currently only 'als' supported

    k : int, optional, default = 5
        Size of low-dimensional basis

    maxiter : int, optional, default = 20
        Maximum number of iterations

    tol : float, optional, default = 0.001
        Tolerance for convergence of iterative algorithm

    h0 : non-negative k x ncols array, optional
        Value at which H is initialized

    w0 : RDD of nrows (tuple, array) pairs, each array of shape (k,), optional, default = None
        Value at which W is initialized

    w_hist : Bool, optional, default = False
        If true, keep track of convergence of w at each iteration

    recon_hist : Bool, optional, default = False
        If true, keep track of reconstruction error at each iteration

    Attributes
    ----------
    `w` : RDD of nrows (tuple, array) pairs, each array of shape (k,)
        Left bases

    `h` : array, shape (k, ncols)
        Right bases

    'h_convergence` : list of floats
        List of Frobenius norms between successive estimates of h

    `w_convergence` : None or list of floats
        If w_hist==True, a list of Frobenius norms between successive estimates of w

    `rec_err` : None or list of floats
        if recon_hist==true, a list of the reconstruction error at each iteration
    """

    def __init__(self, k=5, method='als', maxiter=20, tol=0.001, h0=None, w0=None, w_hist=False, recon_hist=False):
        # initialize input variables
        self.k = int(k)
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.h0 = h0
        self.w0 = w0

        # initialize output variables
        self.h = None
        self.w = None
        self.h_convergence = list()

        if w_hist==True:
            self.w_convergence = list()
        else:
            self.w_convergence = None

        if recon_hist==True:
            self.rec_err = list()
        else:
            self.rec_err = None

    def calc(self, data):
        """
        Calcuate the non-negative matrix decomposition

        Parameters
        ----------
        mat : RDD of (tuple, array) pairs, or RowMatrix
            Matrix to compute non-negative bases from

        Returns
        ----------
        self : returns an instance of self.
        """

        # a a helper function generate a random vector by setting a unique seed
        def randomVector(key, seed, k):
            if not np.iterable(key):
                key = [key]

            # input checking
            assert(np.iterable(key))
            assert(np.iterable(seed))

            #  create unique key
            uniqueKey = list(key)
            uniqueKey.extend(seed)
            np.random.seed(uniqueKey)

            # generate random output
            return np.random.rand(k)

        # a helper function to take the Frobenius norm of two zippable RDDs
        def rddFrobeniusNorm(A, B):
            return np.sqrt(A.zip(B).map(lambda ((keyA, x), (keyB, y)): sum((x - y) ** 2)).reduce(np.add))

        # input checking
        k = self.k
        if k < 1:
            raise ValueError("Supplied k must be greater than 1.")
        m = data.values().first().size
        if self.h0 is not None:
            if np.any(self.h0 < 0):
                raise ValueError("Supplied h0 contains negative entries.")
        if self.w0 is not None:
            if data.map(lambda (k, v): np.any(v < 0)).reduce(np.logical_or):
                raise ValueError("Supplied w0 contains negative entries.")

        # alternating least-squares implementation
        if self.method == "als":

            # initialize NMF and begin als algorithm
            print "Initializing NMF"
            iter = 0
            h_conv_curr = 100
            rec_err = 100

            if self.h0 is None:
                self.h0 = np.random.rand(k, m)
            if self.w0 is None:
                seed = np.ceil(np.random.rand(1)*1e8)
                self.w0 = data.map(lambda (key, _): (key, randomVector(key, seed, k)))

            h = self.h0
            w = self.w0

            # goal is to solve R = WH subject to all entries of W,H >= 0
            # by iteratively updating W and H with least squares and clipping negative values
            while (iter < self.maxiter) and (h_conv_curr > self.tol):
                # update values on iteration
                h_old = h
                w_old = w

                # precompute inv(W' * W) to get inv_gramian_w, a np array
                # We have chosen k such that rank(W) = k, so W'*W should be invertible
                gramian_w = w.values().map(lambda x: np.outer(x, x)).reduce(np.add)
                inv_gramian_w = np.linalg.inv(gramian_w)

                # pseudoinverse of W is inv(W' * W) * W' = inv_gramian_w * w
                pinv_w = w.mapValues(lambda x: np.dot(inv_gramian_w, x))

                # update H using least squares row-wise with inv(W' * W) * W * R (same as pinv(W) * R)
                h = pinv_w.values().zip(data.values()).map(lambda (x, y): np.outer(x, y)).reduce(np.add)

                # clip negative values of H
                h = np.maximum(h, 0)

                # normalize the rows of H for interpretability
                h = np.dot(np.diag(1 / np.maximum(np.linalg.norm(h, axis=1), 0.001)), h)

                # precompute pinv(H) = inv(H' x H) * H' (easy here because h is an np array)
                # the rows of H should be a basis of dimension k, so in principle we could just compute directly
                pinv_h = np.linalg.pinv(h)

                # update W using least squares row-wise with R * pinv(H); then clip negative values to 0
                w = data.mapValues(lambda x: np.dot(x, pinv_h))

                # clip negative values of W
                w = w.mapValues(lambda x: np.maximum(x, 0))

                # estimate convergence
                h_conv_curr = np.linalg.norm(h-h_old)
                self.h_convergence.append(h_conv_curr)
                if self.w_convergence is not None:
                    self.w_convergence.append(rddFrobeniusNorm(w, w_old))

                # calculate reconstruction error
                if self.rec_err is not None:
                    rec_data = w.mapValues(lambda x: np.dot(x, h))
                    self.rec_err.append(rddFrobeniusNorm(data, rec_data))

                # report progress
                print "finished als iteration %d with convergence = %.6f in H" % (iter, h_conv_curr)

                # increment count
                iter += 1

            # report on convergence
            if h_conv_curr <= self.tol:
                print "Converged to specified tolerance."
            else:
                print "Warning: reached maxiter without converging to specified tolerance."

            # report results
            self.h = h
            self.w = w

        else:
            print "Error: %s is not a supported algorithm." % self.method

        return self


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do non-negative matrix factorization")
    parser.add_argument("master", type=str)
    parser.add_argument("datafile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("k", type=int)
    parser.add_argument("--nmfmethod", choices="als", default="als", required=False)
    parser.add_argument("--maxiter", type=float, default=20, required=False)
    parser.add_argument("--tol", type=float, default=0.001, required=False)
    parser.add_argument("--w_hist", type=bool, default=False, required=False)
    parser.add_argument("--recon_hist", type=bool, default=False, required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "dff-highpass", "sub", "dff-percentile"),
                        default="dff-percentile", required=False)

    args = parser.parse_args()

    sc = SparkContext(args.master, "nmf")

    if args.master != "local":
        egg = glob.glob(os.path.join(os.environ['THUNDER_EGG'], "*.egg"))
        sc.addPyFile(egg[0])

    data = load(sc, args.datafile, args.preprocess).cache()
    nmf = NMF(k=args.k, method=args.nmfmethod, maxiter=args.maxiter, tol=args.tol, w_hist=args.w_hist,
              recon_hist=args.recon_hist)
    nmf.calc(data)

    outputdir = args.outputdir + "-nmf"
    save(nmf.w, outputdir, "w", "matlab")
    save(nmf.h, outputdir, "h", "matlab")
    if args.w_hist:
        save(nmf.w_convergence, outputdir, "w_convergence", "matlab")
    if args.recon_hist:
        save(nmf.rec_err, outputdir, "rec_err", "matlab")