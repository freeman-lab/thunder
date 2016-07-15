from ..base import Base
import logging


class Blocks(Base):
    """
    Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Base._metadata + ['blockshape']

    def __init__(self, values):
        super(Blocks, self).__init__(values)

    @property
    def _constructor(self):
        return Blocks

    @property
    def blockshape(self):
        if self.mode == 'spark':
            return tuple(self.values.plan)

        if self.mode == 'local':
            return tuple(self.values.shape)

    def count(self):
        """
        Explicit count of the number of items.

        For lazy or distributed data, will force a computation.
        """
        if self.mode == 'spark':
            return self.tordd().count()

        if self.mode == 'local':
            return 1

    def map(self, func, dims=None, dtype=None):
        """
        Apply an array -> array function to each block
        """
        if self.mode == 'spark':
            mapped = self.values.map(func, value_shape=dims, dtype=dtype)

        if self.mode == 'local':
            if dims is not None:
                logger = logging.getLogger('thunder')
                logger.warn("dims has no meaning in Blocks.map in local mode")
            mapped = func(self.values)

        return self._constructor(mapped).__finalize__(self, noprop=('dtype',))

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'spark':
            return self.values.tordd().values().first()

        if self.mode == 'local':
            return self.values

    def toimages(self):
        """
        Convert blocks to images.
        """
        from thunder.images.images import Images

        if self.mode == 'spark':
            values = self.values.values_to_keys((0,)).unchunk()

        if self.mode == 'local':
            values = self.values

        return Images(values)

    def toseries(self):
        """
        Converts blocks to series.
        """
        from thunder.series.series import Series

        if self.mode == 'spark':
            values = self.values.values_to_keys(tuple(range(1, len(self.shape)))).unchunk()

        if self.mode == 'local':
            n = len(self.shape) - 1
            values = self.values.transpose(tuple(range(1, n+1)) + (0,))

        return Series(values)
