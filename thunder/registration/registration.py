from thunder.rdds.images import Images
from thunder.utils.common import checkParams
from thunder.utils.serializable import Serializable


class Registration(object):
    """
    Class for constructing registration algorithms.

    Construct a registration algorithm by specifying its name,
    and passing any additional keyword arguments. The algorithm
    can then be used to fit registration parameters on an Images object.

    Parameters
    ----------
    method : string
        A registration method, options include 'crosscorr' and 'planarcrosscorr'
    """

    def __new__(cls, method, **kwargs):

        from thunder.registration.methods.crosscorr import CrossCorr, PlanarCrossCorr
        from thunder.registration.methods.lucaskanade import LucasKanade

        REGMETHODS = {
            'crosscorr': CrossCorr,
            'planarcrosscorr': PlanarCrossCorr,
            'lucaskanade': LucasKanade,
        }

        checkParams(method, REGMETHODS.keys())

        return REGMETHODS[method](**kwargs)

    @staticmethod
    def load(file):
        """
        Load a registration model from a file specified using JSON.

        See also
        --------
        RegistrationModel.save : specification for saving registration models
        RegistrationModel.load : specifications for loading registration models
        """

        return RegistrationModel.load(file)


class RegistrationMethod(object):
    """
    Base class for registration methods
    """

    def __init__(self, *args, **kwargs):
        pass

    def prepare(self, **kwargs):
        raise NotImplementedError

    def isPrepared(self, images):
        raise NotImplementedError

    def getTransform(self, im):
        raise NotImplementedError

    def fit(self, images):
        """
        Compute registration parameters on a collection of images / volumes.

        Will return the estimated registration parameters to the driver in the form
        of a RegisterModel, which can then be used to transform Images data.

        Parameters
        ----------
        images : Images
            An Images object with the images / volumes to estimate registration for.

        Returns
        -------
        model : RegisterModel
            Registration params as a model that can be used to transform an Images object.

        See also
        --------
        RegistrationModel : model for applying transformations
        """

        if len(images.dims.count) not in set([2, 3]):
                raise Exception('Number of image dimensions %s must be 2 or 3' % (len(images.dims.count)))

        self.isPrepared(images)

        # broadcast the registration model
        bcReg = images.rdd.context.broadcast(self)

        # compute the transformations
        transformations = images.rdd.mapValues(lambda im: bcReg.value.getTransform(im)).collectAsMap()

        # construct the model
        regMethod = self.__class__.__name__
        transClass = transformations.itervalues().next().__class__.__name__
        model = RegistrationModel(transformations, regMethod=regMethod, transClass=transClass)
        return model

    def run(self, images):
        """
        Compute and implement registration on a collection of images / volumes.

        This is a lazy operation that combines the estimation of registration
        with its implementaiton. It returns a new Images object with transformed
        images, and does not expose the registration parameters directly, see the
        'fit' method to obtain parameters directly.

        Parameters
        ----------
        images : Images
            An Images object with the images / volumes to apply registration to.

        Return
        ------
        Images object with registered images / volumes
        """

        if not (isinstance(images, Images)):
            raise Exception('Input data must be Images or a subclass')

        if len(images.dims.count) not in set([2, 3]):
            raise Exception('Number of image dimensions %s must be 2 or 3' % (len(images.dims.count)))

        self.isPrepared(images)

        # broadcast the reference
        bcReg = images.rdd.context.broadcast(self)

        def fitandtransform(im, reg):
            t = reg.value.getTransform(im)
            return t.apply(im)

        newrdd = images.rdd.mapValues(lambda im: fitandtransform(im, bcReg))

        return Images(newrdd).__finalize__(images)


class RegistrationModel(Serializable, object):

    def __init__(self, transformations, regMethod=None, transClass=None):
        self.transformations = transformations
        self.regMethod = regMethod
        self.transClass = transClass

    def __getitem__(self, entry):
        if not isinstance(entry, int):
            raise IndexError("Selection not recognized, must be Int, got %s" % type(entry))
        return self.transformations[entry]

    def toArray(self):
        """
        Return transformations as an array with shape (n,x1,x2,...)
        where n is the number of images, and remaining dimensions depend
        on the particular transformations
        """
        from numpy import asarray
        collected = [x.toArray() for x in self.transformations.values()]
        return asarray(collected)

    def transform(self, images):
        """
        Apply the transformation to an Images object.

        Will apply the underlying dictionary of transformations to
        the images or volumes of the Images object. The dictionary acts as a lookup
        table specifying which transformation should be applied to which record of the
        Images object based on the key. Because transformations are small,
        we broadcast the transformations rather than using a join.
        """

        from thunder.rdds.images import Images

        # broadcast the transformations
        bcTransformations = images.rdd.context.broadcast(self.transformations)

        # apply the transformations
        newrdd = images.rdd.map(lambda (k, im): (k, bcTransformations.value[k].apply(im)))
        return Images(newrdd).__finalize__(images)

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%g transformations' % (len(self.transformations))
        s += '\nregistration method: ' + self.regMethod
        s += '\ntransformation type: ' + self.transClass
        return s




