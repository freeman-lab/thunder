"""
Utilities for saving data
"""

import os
from scipy.io import savemat
from numpy import std
from math import isnan
from numpy import array, squeeze, sum, shape, reshape, transpose, maximum, minimum, float16, uint8, savetxt, size, arange
from matplotlib.pyplot import imsave
from matplotlib import cm
from thunder.utils.load import getdims, subtoind, isrdd, Dimensions


def arraytoim(mat, filename, format="png"):
    """Write a 2D numpy array to a grayscale image. If mat is 3D,
    will separately write each image along the 3rd dimension

    Parameters
    ----------
    mat : array (2D or 3D), dtype must be uint8
        Pixel values for image or set of images to write

    filename : str
        Base filename for writing

    format : str, optional, default = "png"
        Image format to write (see matplotlib's imsave for options)
    """
    dims = shape(mat)
    if len(dims) > 2:
        for z in range(0, dims[2]):
            cdata = mat[:, :, z]
            imsave(filename+"-"+str(z)+"."+format, cdata, cmap=cm.gray)
    elif len(dims) == 2:
        imsave(filename+"."+format, mat, cmap=cm.gray)
    else:
        raise NotImplementedError('array must be 2 or 3 dimensions for image writing')


def rescale(data):
    """Rescale data to lie between 0 and 255 and convert to uint8

    For strictly positive data, subtract the min and divide by max
    otherwise, divide by the maximum absolute value and center

    If each element of data has multiple entries,
    they will be rescaled separately
    """
    if size(data.first()[1]) > 1:
        data = data.mapValues(lambda x: map(lambda y: 0 if isnan(y) else y, x))
    else:
        data = data.mapValues(lambda x: 0 if isnan(x) else x)
    mnvals = data.map(lambda (_, v): v).reduce(minimum)
    mxvals = data.map(lambda (_, v): v).reduce(maximum)
    if sum(mnvals < 0) == 0:
        data = data.mapValues(lambda x: uint8(255 * (x - mnvals)/(mxvals - mnvals)))
    else:
        mxvals = maximum(abs(mxvals), abs(mnvals))
        data = data.mapValues(lambda x: uint8(255 * ((x / (2 * mxvals)) + 0.5)))
    return data


def subset(data, nsamples=100, thresh=None):
    """Extract subset of points from an RDD into a local array,
    filtering on the standard deviation

    Parameters
    ----------
    data : RDD of (tuple, array) pairs
        The data to get a subset from

    nsamples : int, optional, default = 100
        The number of data points to sample

    thresh : float, optional, default = None
        A threshold on standard deviation to use when picking points

    Returns
    -------
    result : array
        A local numpy array with the subset of points
    """
    if thresh is not None:
        result = array(data.values().filter(lambda x: std(x) > thresh).takeSample(False, nsamples))
    else:
        result = array(data.values().takeSample(False, nsamples))
    return result


def pack(data, ind=None, dims=None, sorting=False, axis=None):
    """Pack an RDD into a dense local array, with options for
    sorting, reshaping, and projecting based on keys

    Parameters
    ----------
    data : RDD of (tuple, array) pairs
        The data to pack into a local array

    ind : int, optional, default = None
        An index, if each record has multiple entries

    dims : Dimensions, optional, default = None
        Dimensions of the keys, for use with sorting and reshaping

    sorting : Boolean, optional, default = False
        Whether to sort the RDD before packing

    axis : int, optional, default = None
        Which axis to do maximum projection along

    Returns
    -------
    result : array
        A local numpy array with the RDD contents

    """

    if dims is None:
        dims = getdims(data)

    if axis is not None:
        nkeys = len(data.first()[0])
        if axis > nkeys - 1:
            raise IndexError('only %g keys, cannot compute maximum along axis %g' % (nkeys, axis))
        data = data.map(lambda (k, v): (tuple(array(k)[arange(0, nkeys) != axis]), v)).reduceByKey(maximum)
        dims.min = list(array(dims.min)[arange(0, nkeys) != axis])
        dims.max = list(array(dims.max)[arange(0, nkeys) != axis])
        sorting = True  # will always need to sort because reduceByKey changes order

    if ind is None:
        result = data.map(lambda (_, v): float16(v)).collect()
        nout = size(result[0])
    else:
        result = data.map(lambda (_, v): float16(v[ind])).collect()
        nout = size(ind)

    if sorting is True:
        data = subtoind(data, dims.max)
        keys = data.map(lambda (k, _): int(k)).collect()
        result = array([v for (k, v) in sorted(zip(keys, result), key=lambda (k, v): k)])

    # reshape into a dense array of shape (b, x, y, z)  or (b, x, y) or (b, x)
    # where b is the number of outputs per record
    out = transpose(reshape(result, ((nout,) + dims.count())[::-1]))

    # flip xy for spatial data
    if size(dims.count()) == 3:  # (b, x, y, z) -> (b, y, x, z)
        out = out.transpose([0, 2, 1, 3])
    if size(dims.count()) == 2:  # (b, x, y) -> (b, y, x)
        out = out.transpose([0, 2, 1])

    return squeeze(out)


def save(data, outputdir, outputfile, outputformat, sorting=False, dimsmax=None, dimsmin=None):
    """
    Save data to a variety of formats
    Automatically determines whether data is an array
    or an RDD and handle appropriately

    Parameters
    ----------
    data : RDD of (tuple, array) pairs, or numpy array
        The data to save

    outputdir : str
        Output directory

    outputfile : str
        Output filename

    outputformat : str
        Output format ("matlab", "text", or "image")
    """

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    filename = os.path.join(outputdir, outputfile)

    if isrdd(data):
        nout = size(data.first()[1])
        if dimsmax is not None:
            dims = Dimensions()
            dims.max = dimsmax
            if dimsmin is not None:
                dims.min = dimsmin
            else:
                dims.min = (1, 1, 1)
        elif dimsmin is not None:
            raise Exception('cannot provide dimsmin without dimsmax')
        else:
            dims = getdims(data)

    if (outputformat == "matlab") | (outputformat == "text"):
        if isrdd(data):
            if nout > 1:
                for iout in range(0, nout):
                    result = pack(data, ind=iout, dims=dims, sorting=sorting)
                    if outputformat == "matlab":
                        savemat(filename+"-"+str(iout)+".mat", mdict={outputfile+str(iout): result},
                                oned_as='column', do_compression='true')
                    if outputformat == "text":
                        savetxt(filename+"-"+str(iout)+".txt", result, fmt="%.6f")
            else:
                result = pack(data, dims=dims, sorting=sorting)
                if outputformat == "matlab":
                    savemat(filename+".mat", mdict={outputfile: result},
                            oned_as='column', do_compression='true')
                if outputformat == "text":
                    savetxt(filename+".txt", result, fmt="%.6f")
        else:
            if outputformat == "matlab":
                savemat(filename+".mat", mdict={outputfile: data}, oned_as='column', do_compression='true')
            if outputformat == "text":
                savetxt(filename+".txt", data, fmt="%.6f")

    if outputformat == "image":
        if isrdd(data):
            data = rescale(data)
            if nout > 1:
                for iout in range(0, nout):
                    result = pack(data, ind=iout, dims=dims, sorting=sorting)
                    arraytoim(result, filename+"-"+str(iout))
            else:
                result = pack(data, dims=dims, sorting=sorting)
                arraytoim(result, filename)
        else:
            arraytoim(data, filename)



