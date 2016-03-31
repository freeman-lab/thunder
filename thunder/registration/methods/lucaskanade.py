""" Registration methods based on Lucas-Kanade registration"""

from numpy import array, ndarray, inf, zeros

from thunder.rdds.images import Images
from thunder.registration.registration import RegistrationMethod
from thunder.registration.transformation import TRANSFORMATION_TYPES
from thunder.registration.methods.utils import volumesToMatrix,  solveLinearized, computeReferenceMean, checkReference


class LucasKanade(RegistrationMethod):
    """Lucas-Kanade registration method.

    Lucas-Kanade (LK) is an iterative algorithm for aligning an image to a reference. It aims to minimize the squared
    error between the transformed image and the reference. As the relationship between transformation parameters and
    transformed pixels is nonlinear, we have to perform a series of linearizations similar to Levenberg-Marquardt.
    At each iteration, we compute the Jacobian of the output image with respect to the input parameters, and solve a
    least squares problem to identify a change in parameters. We update the parameters and repeat.

    To increase robustness, we extend the traditional LK algorithm to use a set of reference images.
    We minimize the squared error of the difference of the transformed image and a learned weighting of the references.
    """

    def __init__(self, transformationType='Translation', border=0, tol=1e-5, maxIter=100, robust=False):
        """
        Parameters
        ----------
        transformationType : one of 'Translation', 'Euclidean', optional, default = 'Translation'
            type of transformation to use
        border : int or tuple, optional, default = 0
            Border to be zeroed out after transformations. For most datasets, it is
            critical that this value be larger than the maximum translation to get
            good results and avoid boundary artifacts.
        maxIter : int, optional, default = 100
            maximum number of iterations
        tol : float, optional, default = 1e-5
            stopping criterion on the L2 norm of the change in parameters
        robust : bool, optional, default = False
            solve a least absolute deviation problem instead of least squares
        """
        self.transformationType = transformationType
        self.border = border
        self.maxIter = maxIter
        self.tol = tol
        self.robust = robust

    def prepare(self, images, startidx=None, stopidx=None):
        """
        Prepare Lucas-Kanade registration by computing or specifying a reference image.

        Parameters
        ----------
        images : ndarray or Images object
            Images to compute reference from, or a single image to set as reference

        See computeReferenceMean.
        """
        if isinstance(images, Images):
            self.reference = computeReferenceMean(images, startidx, stopidx)
        elif isinstance(images, ndarray):
            self.reference = images
        else:
            raise Exception('Must provide either an Images object or a reference')
        # Convert references to matrix to speed up solving linearized system
        self.referenceMat = volumesToMatrix(self.reference)
        return self

    def isPrepared(self, images):
        """
        Check if Lucas-Kanade is prepared by checking the dimensions of the reference.

        See checkReference.
        """

        if not hasattr(self, 'reference'):
            raise Exception('Reference not defined')
        else:
            checkReference(self.reference, images)

    def getTransform(self, vol):
        from numpy.linalg import norm
        # Create initial transformation
        tfm = TRANSFORMATION_TYPES[self.transformationType](shift=zeros(vol.ndim))
        iter = 0
        normDelta = inf
        params = []
        while iter < self.maxIter and normDelta > self.tol:
            volTfm, jacobian = tfm.jacobian(vol, border=self.border)
            deltaTransformParams, coeff = solveLinearized(volumesToMatrix(volTfm), volumesToMatrix(jacobian), self.referenceMat, self.robust)
            tfm.updateParams(deltaTransformParams)
            normDelta = norm(deltaTransformParams)
            params.append(tfm.getParams())
            iter += 1
        return tfm
