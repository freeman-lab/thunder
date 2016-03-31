""" Shared utilities for registration methods """

from numpy import ndarray

from thunder.rdds.images import Images


def computeReferenceMean(images, startIdx=None, stopIdx=None, defaultNImages=20):
    """
    Compute a reference by taking the mean across images.

    The default behavior is to take the mean across the center `defaultNImages` records
    in the Images RDD. If startIdx or stopIdx is specified, then the mean will be
    calculated across this range instead.

    Parameters
    ----------
    images : Images
            An Images object containing the image / volumes to compute reference from

    startIdx : int, optional, default = None
        Starting index if computing a mean over a specified range

    stopIdx : int, optional, default = None
        Stopping index (exclusive) if computing a mean over a specified range

    defaultNImages : int, optional, default = 20
        Number of images across which to calculate the mean if neither startIdx nor stopIdx
        is given.

    Returns
    -------
    refval : ndarray
        The reference image / volume
    """

    if not (isinstance(images, Images)):
        raise Exception('Input data must be Images or a subclass')

    doFilter = True
    if startIdx is None and stopIdx is None:
        n = images.nrecords
        if n <= defaultNImages:
            doFilter = False
        else:
            ctrIdx = n / 2  # integer division
            halfWindow = defaultNImages / 2  # integer division
            parity = 1 if defaultNImages % 2 else 0
            startIdx = ctrIdx - halfWindow
            stopIdx = ctrIdx + halfWindow + parity
            n = stopIdx - startIdx
    else:
        if startIdx is None:
            startIdx = 0
        if stopIdx is None:
            stopIdx = images.nrecords
        n = stopIdx - startIdx

    if doFilter:
        rangePredicate = lambda x: startIdx <= x < stopIdx
        ref = images.filterOnKeys(rangePredicate)
    else:
        ref = images

    reference = (ref.sum() / float(n)).astype(images.dtype)

    return reference


def checkReference(images, reference):
    """
    Check that a reference is an ndarray and matches the dimensions of images.

    Parameters
    ----------
    images : Images
        An Images object containing the image / volumes to check against the reference

    reference : ndarray
        A reference image / volume
    """

    if isinstance(reference, ndarray):
        if reference.shape != images.dims.count:
            raise Exception('Dimensions of reference %s do not match dimensions of data %s' %
                            (reference.shape, images.dims.count))
        else:
            raise Exception('Reference must be an array')


def computeDisplacement(arry1, arry2):
    """
    Compute an optimal displacement between two ndarrays.

    Finds the displacement between two ndimensional arrays. Arrays must be
    of the same size. Algorithm uses a cross correlation, computed efficiently
    through an n-dimensional fft.

    Parameters
    ----------
    arry1 : ndarray
        The first array

    arry2 : ndarray
        The second array
    """

    from numpy.fft import rfftn, irfftn
    from numpy import unravel_index, argmax

    # compute real-valued cross-correlation in fourier domain
    s = arry1.shape
    f = rfftn(arry1)
    f *= rfftn(arry2).conjugate()
    c = abs(irfftn(f, s))

    # find location of maximum
    inds = unravel_index(argmax(c), s)

    # fix displacements that are greater than half the total size
    pairs = zip(inds, arry1.shape)
    # cast to basic python int for serialization
    adjusted = [int(d - n) if d > n // 2 else int(d) for (d, n) in pairs]

    return adjusted

def zeroBorder(vol, border=1, cval=0.0):
    """Zero-out boundary voxels of a volume.

    Parameters
    ----------
        vol: 3d volume
        border: scalar or tuple containing borders for each dimension
        cval: constant value to replace boundary voxels with, default is 0.0
    """
    import numpy as np # sorry
    vol = vol.copy() # create new copy so we don't overwrite vol
    dims = np.array(vol.shape)
    if np.size(border) == 1:
        border = border * np.ones(vol.ndim, dtype=int)
    border = np.array(border, dtype=int)
    # Don't apply border to singleton dimensions.
    border[dims == 1] = 0
    assert len(border) == vol.ndim
    if np.any(dims - border <= 0):
        raise ValueError('Border %s exceeds volume shape %s.' %
                (str(border), str(dims)) )
    for dim, bval in enumerate(border):
        if bval > 0:
            slices = [slice(bval) if d == dim else slice(None) for d in xrange(vol.ndim)]
            vol[slices] = cval
            slices[dim] = slice(-bval, None)
            vol[slices] = cval
    return vol


def solveLinearized(vec, jacobian, reference, robust=False):
    """Solve linearized registration problem for change in transformation parameters and weighting on reference basis.

    Parameters
    ----------
    vec : array, shape (nvoxels,)
        vectorized volume
    jacobian: array, shape (nvoxels, nparams)
        jacobian for each transformation parameter
    reference: array, shape (nvoxels, nbasis)
        array of vectorized reference volumes
    robust: bool, optional, default = False
        solve a least absolute deviation problem instead of least squares

    Returns
    -------
    deltaTransforms : array, shape (nparams,)
        optimal change in transformation parameters
    coeff : array, shape (nbasis,)
        optimal weighting of reference volumes
    """
    from numpy import column_stack
    A = column_stack((jacobian, reference))
    if robust:
        from statsmodels.regression.quantile_regression import QuantReg
        quantile = 0.5
        model = QuantReg(vec, A).fit(q=quantile)
        params = model.params
    else:
        from numpy.linalg import lstsq
        model = lstsq(A, vec)
        params = model[0]
    from numpy import split
    deltaTransform, coeff = split(params, [jacobian.shape[1]])
    return deltaTransform, coeff


def volToVec(vol):
    """
    Convert volume to vector.

    Parameters
    ----------
    vol : array
        volume

    Returns
    -------
    vectorized volume
    """
    return vol.ravel()


def vecToVol(vec, dims):
    """
    Convert vector to volume.

    Parameters
    ----------
    vec : array, shape (nvoxels,)
        vectorized volume
    dims : tuple, optional
        shape of volume, must be set to reshape vectors into volumes

    Returns
    -------
    volume if input is a vector, and vector if input is a volume
    """
    assert vec.size == prod(dims)
    return vec.reshape(dims)


def volumesToMatrix(vols):
    """Convert list of volumes to a matrix.

    Parameters
    ----------
    vols : list of arrays

    Returns
    -------
    array with size nvoxels by number of volumes
    """
    from numpy import column_stack
    if not isinstance(vols, list):
        return volToVec(vols)
    else:
        return column_stack(map(volToVec, vols)).squeeze()


def imageGradients(im, sigma=None):
    """Compute gradients of volume in each dimension using a Sobel filter.

    Parameters
    ----------
    im : ndarray
        single volume
    sigma : float or tuple, optional, default = None
        smoothing amount to apply to volume before computing gradients
    """
    from scipy.ndimage.filters import gaussian_filter, sobel
    if sigma is not None:
        im = gaussian_filter(im, sigma)
    grads = [sobel(im, axis=dim, mode='constant') / 8.0 for dim in xrange(im.ndim)]
    return grads

