from numpy import prod, rollaxis

from ..base import Base
import logging


class Blocks(Base):
    """
    Superclass for subdivisions of Images data.

    Subclasses of Blocks will be returned by an images.toBlocks() call.
    """
    _metadata = Base._metadata + ['blockshape', 'padding']

    def __init__(self, values):
        super(Blocks, self).__init__(values)

    @property
    def _constructor(self):
        return Blocks

    @property
    def blockshape(self):
        return tuple(self.values.plan)

    @property
    def padding(self):
        return tuple(self.values.padding)

    def count(self):
        """
        Explicit count of the number of items.

        For lazy or distributed data, will force a computation.
        """
        if self.mode == 'spark':
            return self.tordd().count()

        if self.mode == 'local':
            return prod(self.values.values.shape)

    def collect_blocks(self):
        """
        Collect the blocks in a list
        """
        if self.mode == 'spark':
            return self.values.tordd().sortByKey().values().collect()

        if self.mode == 'local':
            return self.values.values.flatten().tolist()

    def map(self, func, value_shape=None, dtype=None):
        """
        Apply an array -> array function to each block
        """
        mapped = self.values.map(func, value_shape=value_shape, dtype=dtype)
        return self._constructor(mapped).__finalize__(self, noprop=('dtype',))

    def map_generic(self, func):
        """
        Apply an arbitrary array -> object function to each blocks.
        """
        return self.values.map_generic(func)[0]

    def first(self):
        """
        Return the first element.
        """
        if self.mode == 'spark':
            return self.values.tordd().values().first()

        if self.mode == 'local':
            return self.values.first

    def toimages(self):
        """
        Convert blocks to images.
        """
        from thunder.images.images import Images

        if self.mode == 'spark':
            values = self.values.values_to_keys((0,)).unchunk()

        if self.mode == 'local':
            values = self.values.unchunk()

        return Images(values)

    def toseries(self):
        """
        Converts blocks to series.
        """
        from thunder.series.series import Series

        if self.mode == 'spark':
            values = self.values.values_to_keys(tuple(range(1, len(self.shape)))).unchunk()

        if self.mode == 'local':
            values = self.values.unchunk()
            values = rollaxis(values, 0, values.ndim)

        return Series(values)

    def toarray(self):
        """
        Convert blocks to local ndarray
        """
        if self.mode == 'spark':
            return self.values.unchunk().toarray()

        if self.mode == 'local':
            return self.values.unchunk()
