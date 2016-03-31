""" Transformations produced by registration methods """
import numpy as np
from numpy import asarray
from thunder.utils.serializable import Serializable


class Transformation(object):
    """ Base class for transformations """

    def apply(self, im):
        raise NotImplementedError

# Module-level dictionary cache mapping volume dimensions to an array containing
# the image coordinates for that volume. These coordinates are used to speed up
# application of transformations that inherit from GridMixin.
_grid = {}

def getGrid(dims):
    """
    Check and (if needed) initialize a grid with the appropriate dimensions.

    Parameters
    ----------
    dims : tuple
        shape of volume
    """
    global _grid
    if dims not in _grid:
        _grid[dims] = np.vstack((np.array(np.meshgrid(*[np.arange(d) for d in dims], indexing='ij')),
                          np.ones(dims)[np.newaxis, :, :]))
        # Prevent grid from being altered.
        _grid[dims].flags.writeable = False

    return _grid[dims]

class GridMixin(object):
    def getCoords(self, grid):
        """
        Get the coordinates where the input volume is evaluated.

        Returns
        -------
        array with size ndims by nvoxels specifying the coordinates for each voxel.
        """
        raise NotImplementedError

    def apply(self, vol, order=1, cval=0.0, **kwargs):
        from scipy.ndimage import map_coordinates
        grid = getGrid(vol.shape)
        coords = self.getCoords(grid)
        tvol = map_coordinates(vol, coords, order=order, cval=0.0, **kwargs)
        return tvol

class DifferentiableTransformation(Transformation):
    """
    Differentiable transformations must have methods to compute the Jacobian and update parameters. These are used by
    iterative alignment techniques like Lucas-Kanade and RASL.
    """

    def _jacobianCoords(self, vol, sigma=None, normalize=True, border=1):
        """Compute Jacobian of volume with respect to the coordinates.

        Args:
            vol: volume
            tfm: Transform object
            sigma: smoothing bandwidth for gradients (None for no smoothing)
            normalize: Whether to normalize images before aligning.
            border: Number or tuple of border sizes to zero after transforming.
        Returns:
            tvol : array
                transformed volume
            grads : list of arrays
                list of volume Jacobians with respect to coordinates, one for each dimension
        """
        from numpy.linalg import norm
        from thunder.registration.methods.utils import imageGradients, zeroBorder

        grads = imageGradients(vol, sigma)
        tvol = zeroBorder(self.apply(vol), border)
        normVol = norm(tvol.ravel())
        if normVol == 0.0:
            raise ValueError('Transform yields volume of zeroes.')
        grads = [zeroBorder(self.apply(grad), border) for grad in grads]
        if normalize:
            if normVol == 0.0:
                raise ValueError('Transform yields volume of zeroes.')
            # Update gradients to reflect normalization
            grads = [grad / normVol - (grad * tvol).sum() / (normVol**3) * tvol for grad in grads]
            tvol /= normVol
        return tvol, grads

    def jacobian(self, vol, **kwargs):
        """
        Compute gradient of transformation output with respect to parameters.

        Parameters
        ----------
        vol : array
            volume

        Returns
        -------
        ordered list of arrays containing the Jacobian for each parameter
        """
        raise NotImplementedError

    def getParams(self):
        """
        Get the parameters of this transformation.

        Returns
        -------
        array with shape (nparams,)
        """
        raise NotImplementedError

    def setParams(self, params):
        """
        Set each parameter in the transformation.

        Parameters
        ----------
        params : array, shape (nparams,)
            new values of parameters. The ordering here must match
            the ordering of the parameters returned by jacobian.
        """
        raise NotImplementedError

    def updateParams(self, deltaParams):
        """
        Update each parameter in the transformation.

        Parameters
        ----------
        deltaParams : array, shape (nparams,)
            ordered changes in parameters to be applied. The ordering here must match
            the ordering of the parameters returned by jacobian.
        """
        self.setParams(self.getParams() + deltaParams)


class TranslationTransformation(DifferentiableTransformation):
    def __init__(self, shift):
        """Translation in 3d.

        Parameters
        ----------
            shift: 3d vector of translations (x, y, z)

        """
        self.shift = shift

    def jacobian(self, vol, **kwargs):
        return self._jacobianCoords(vol, **kwargs)

    def getParams(self):
        return np.asarray(self.shift)

    def setParams(self, shift):
        assert len(shift) == len(self.shift)
        self.shift = shift

    def apply(self, vol):
        from scipy.ndimage.interpolation import shift
        return shift(vol, self.shift, mode='constant', cval=0.0)


class ProjectiveTransformation(GridMixin, DifferentiableTransformation):
    """
    Projective transformations are differentiable and can be represented as a matrix."""

    def __init__(self, center=None):
        self.center = center

    def asMatrix(self):
        raise NotImplementedError

    def getCoords(self, grid):
        d = grid.shape[0] - 1  # number of dimensions
        dims = grid.shape[1:]
        if self.center is None:
            self.center = (np.array(dims) - 1) / 2.0

        A = self.asMatrix()

        # Center grid so that we apply rotations w.r.t. given center
        center_bcast = np.r_[self.center, 0].reshape((-1, ) + (1,) * d)
        if self.center is not None:
            grid = grid - center_bcast

        # Apply transformation
        grid = np.tensordot(A.T, grid, axes=(0,0))

        # Move back to voxel reference frame
        if self.center is not None:
            grid = grid + center_bcast
        return grid[:-1]  # throw out last homogeneous coordinate

class EuclideanTransformation(ProjectiveTransformation):
    def __init__(self, shift, rotation=None, zTranslation=False, zRotation=False, center=None):
        """Translation and rotation in 3d.

        Parameters
        ----------
        shift : list or array
            spatial shifts for each dimension
        rotation : float, list, or array
            rotation in x-y if scalar. rotation in x-y plane, x-z plane, y-z plane if list
            or array and dataset is in 3d.
        zTranslation : bool, optional, default = False
            whether to allow translation in z
        zRotation : bool, optional, default = False
            whether to allow rotation in z
        center : list or array, optional, default = None
            Set center coordinates that define the point around which the volume is rotated.
            Coordinates should be in terms of zero-indexed pixels. For example if the volume was 5x5x3,
            the default center of rotation would be at the center of the volume: (2, 2, 1).
        """
        self.shift = np.atleast_1d(shift)
        self.ndim = len(self.shift)
        if rotation is None:
            if self.ndim == 2 or not zRotation:
                rotation = 0.0
            else:
                rotation = np.zeros(3)
        self.rotation = np.atleast_1d(rotation)
        self.zRotation = zRotation
        self.zTranslation = zTranslation
        self.center = center

    def getParams(self):
        return np.r_[self.shift, self.rotation]

    def updateParams(self, deltaParams):
        if self.ndim == 2:
            self.shift += deltaParams[:2]
            self.rotation += deltaParams[2]
        else:
            self.shift += deltaParams[:3]
            self.rotation += deltaParams[3:]

    def jacobian(self, vol, **kwargs):

        tvol, imageGradients = self._jacobianCoords(vol, **kwargs)
        imageGrid = getGrid(vol.shape)
        ndim = len(self.shift)

        stheta = np.sin(self.rotation[0])
        ctheta = np.cos(self.rotation[0])

        if ndim == 2 or not self.zRotation:
            dtheta = imageGradients[0] * (-imageGrid[0] * stheta  + imageGrid[1] * -ctheta)
            dtheta += imageGradients[1] * ( imageGrid[0] * ctheta  + imageGrid[1] * -stheta)
            dangles = [dtheta]
        else:
            sphi = np.sin(self.rotation[1])
            cphi = np.cos(self.rotation[1])
            spsi = np.sin(self.rotation[2])
            cpsi = np.cos(self.rotation[2])
            dtheta = imageGradients[0] * (
               -imageGrid[0] * cphi * stheta +
                imageGrid[1] * (-ctheta * cpsi - stheta * sphi * spsi) +
                imageGrid[2] * (-cpsi * stheta * sphi + ctheta * spsi))
            dtheta +=  imageGradients[1] * (
                imageGrid[0] * ctheta * cphi +
                imageGrid[1] * (-cpsi * stheta + ctheta * sphi * spsi) +
                imageGrid[2] * (ctheta * cpsi * sphi + stheta * spsi))
            dphi =  imageGradients[0] * (
                imageGrid[2] * ctheta * cphi * cpsi -
                imageGrid[0] * ctheta * sphi +
                imageGrid[1] * ctheta * cphi * spsi)
            dphi += imageGradients[1] * (
                imageGrid[2] * cphi * cpsi * stheta -
                imageGrid[0] * stheta * sphi +
                imageGrid[1] * cphi * stheta * spsi)
            dphi += imageGradients[2] * (
               -imageGrid[0] * cphi -
                imageGrid[2] * cpsi * sphi -
                imageGrid[1] * sphi * spsi)
            dpsi = imageGradients[0] * (
                imageGrid[1] * (ctheta * cpsi * sphi + stheta * spsi) +
                imageGrid[2] * (cpsi * stheta - ctheta * sphi * spsi) )
            dpsi += imageGradients[1] * (
                imageGrid[1] * (cpsi * stheta * sphi - ctheta * spsi) +
                imageGrid[2] * (-ctheta * cpsi - stheta * sphi * spsi))
            dpsi += imageGradients[2] * (imageGrid[1] * cphi * cpsi - imageGrid[2] * cphi * spsi)
            dangles = [dtheta, dphi, dpsi]

        # Zero-out Jacobian corresponding to z translation
        if ndim == 3 and not self.zTranslation:
            imageGradients[2][:] = 0.0
        # Coordinate frame is somehow flipped??

        return tvol, imageGradients + dangles

    def asMatrix(self):
        #XXX: grid coordinate frame is flipped
        return transformationMatrix(-self.shift,  -self.rotation)

    def __repr__(self):
        return "EuclideanTransformation(shift=%s, rotation=%s)" % (repr(self.shift), repr(self.rotation))



class Displacement(Transformation, Serializable):
    """
    Class for transformations based on spatial displacements.

    Can be applied to either images or volumes.

    Parameters
    ----------
    delta : list
        A list of spatial displacements for each dimensino,
        e.g. [10,5,2] for a displacement of 10 in x, 5 in y, 2 in z
    """

    def __init__(self, delta=None):
        self.delta = delta

    def toArray(self):
        """
        Return transformation as an array
        """
        return asarray(self.delta)

    def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        return shift(im, map(lambda x: -x, self.delta), mode='nearest')

    def __repr__(self):
        return "Displacement(delta=%s)" % repr(self.delta)


class PlanarDisplacement(Transformation, Serializable):
    """
    Class for transformations based on two-dimensional spatial displacements.

    Applied separately to each plane of a three-dimensional volume.

    Parameters
    ----------
    delta : list
        A nested list, where the first list is over planes, and
        for each plane a list of [x,y] displacements
    """

    def __init__(self, delta=None):
        self.delta = delta

    def toArray(self):
        """
        Return transformation as an array
        """
        return asarray(self.delta)

    def apply(self, im):
        """
        Apply an 2D displacement by shifting each plane of a volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        if im.ndim == 2:
            return shift(im,  map(lambda x: -x, self.delta[0]))
        else:
            im.setflags(write=True)
            for z in range(0, im.shape[2]):
                im[:, :, z] = shift(im[:, :, z],  map(lambda x: -x, self.delta[z]), mode='nearest')
            return im

    def __repr__(self):
        return "PlanarDisplacement(delta=%s)" % repr(self.delta)


def transformationMatrix(shift, rot=None):
    """Create an affine transformation matrix

    Parameters
    ----------
    shift : array, shape (ndim,)
        translations along each dimension
    rot : scalar or array with shape (ndim, )

    Returns
    -------
    A : array, shape (ndims + 1, ndims + 1)
        transformation matrix that shifts and rotates a set of points in homogeneous coordinates
    """

    ndim = len(shift)
    if rot is None:
        rot = np.zeros(ndim)
    else:
        rot = np.atleast_1d(rot)
    c = np.cos(rot)
    s = np.sin(rot)
    trans = np.eye(ndim + 1)
    trans[:-1, -1] = shift
    xrot = np.array(
            [[c[0], -s[0],  0,    0],
            [s[0],  c[0],  0,    0],
            [0,     0,     1,    0],
            [0,     0,     0,    1]])
    if ndim == 2 or np.size(rot) == 1:
        A = np.dot(trans, xrot[:ndim + 1, :ndim + 1])
    else:
        yrot = [[c[1],  0,     s[1], 0,],
                [0,     1,     0,    0],
                [-s[1], 0,     c[1], 0],
                [0,     0,     0,    1]]
        zrot = [[1,     0,     0,    0],
                [0,     c[2], -s[2], 0],
                [0,     s[2],  c[2], 0],
                [0,     0,     0,    1]]
        A = np.dot(trans, np.dot(xrot, np.dot(yrot, zrot)))
    return A

# Dict of valid types of Transformations used by Lucas-Kanade
TRANSFORMATION_TYPES = {
    'Translation': TranslationTransformation,
    'Euclidean': EuclideanTransformation
}
