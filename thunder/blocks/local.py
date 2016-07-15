from numpy import arange, r_, empty, zeros, random, where, prod, array, asarray, floor
from itertools import product
from bolt.utils import allstack

from ..base import Base
import logging


class LocalChunks:
    """
    Simple helper class for a local chunked ndarray.
    """

    def __init__(self, values, shape, plan, dtype=None, padding=None):
        """
        Create a chunked ndarray.

        Parameters
        ----------
        values : ndarray (dtype 'object') or ndarrays
            Array containing all the chunks.

        shape : tuple
            Shape of the full unchunked array.

        plan : tuple
            Shape of each chunk (excluding edge effects).

        dtype : numpy dtype, optional, default = None
            dtype of chunks. If not given, will be inferred from first chunk.

        padding: tuple, optional, default = None
            amount of padding in each dimension to include in each block
        """
        self.values = values
        self.shape = shape
        self.plan = plan
        self.dtype = dtype
        self.padding = padding

        if self.dtype is None:
            self.dtype = self.first.dtype

        if self.padding is None:
            self.padding = len(self.shape)*(0,)

    @property
    def first(self):
        """
        First chunk
        """
        return self.values[tuple(zeros(len(self.values.shape)))]

    def unchunk(self):
        """
        Reconstitute the chunked array back into a full ndarray.

        Returns
        -------
        ndarray
        """
        if self.padding != len(self.shape)*(0,):
            shape = self.values.shape
            arr = empty(shape, dtype=object)
            for inds in product(*[arange(s) for s in shape]):
                slices = []
                for i, p, n in zip(inds, self.padding, shape):
                    start = None if (i == 0 or p == 0) else p
                    stop = None if (i == n-1 or p == 0) else -p
                    slices.append(slice(start, stop, None))
                arr[inds] = self.values[inds][tuple(slices)]
        else:
            arr = self.values

        return allstack(arr.tolist())

    def map(self, func, value_shape=None, dtype=None):

        if value_shape is None or dtype is None:
            # try to compute the size of each mapped element by applying func to a random array
            try:
                mapped = func(random.randn(*self.plan).astype(self.dtype))
            # if this fails, try to use the first block instead
            except Exception:
                mapped = func(self.first)
            if value_shape is None:
                value_shape = mapped.shape
            if dtype is None:
                dtype = mapped.dtype

        blocked_dims = where(array(self.plan) != array(self.shape))[0]
        unblocked_dims = where(array(self.plan) == array(self.shape))[0]

        # check that no dimensions are dropped
        if len(value_shape) != len(self.plan):
            raise NotImplementedError('map on chunked array cannot drop dimensions')

        # check that chunked dimensions did not change shape
        if any([value_shape[i] != self.plan[i] for i in blocked_dims]):
            raise ValueError('map cannot change the sizes of chunked dimensions')

        newshape = [value_shape[i] if i in unblocked_dims else self.shape[i] for i in range(self.values.ndim)]
        newshape = tuple(newshape)

        c = prod(self.values.shape)
        mapped = empty(c, dtype=object)
        for i, a in enumerate(self.values.flatten()):
            mapped[i] = func(a)
        return LocalChunks(mapped.reshape(self.values.shape), newshape, value_shape, dtype)

    def map_generic(self, func):

        shape = self.values.shape
        n = prod(shape)
        arr = empty(n, dtype=object)
        for i, x in enumerate(self.values.flatten()):
            arr[i] = func(x)
        return arr.reshape(shape)

    @staticmethod
    def chunk(arr, chunk_size="150", padding=None):
        """
        Created a chunked array from a full array and a chunk size.

        Parameters
        ----------
        array : ndarray
            Array that will be broken into chunks

        chunk_size : string or tuple, default = '150'
            Size of each image chunk.
            If a str, size of memory footprint in KB.
            If a tuple, then the dimensions of each chunk.
            If an int, then all dimensions will use this number

        padding : tuple or int
            Amount of padding along each dimensions for chunks. If an int, then
            the same amount of padding is used for all dimensions

        Returns
        -------
        LocalChunks
        """

        plan, _ = LocalChunks.getplan(chunk_size, arr.shape[1:], arr.dtype)
        plan = r_[arr.shape[0], plan]

        if padding is None:
            pad = arr.ndim*(0,)
        elif isinstance(padding, int):
            pad = (0,) + (arr.ndim-1)*(padding,)
        else:
            pad = (0,) + padding

        shape = arr.shape

        if any([x + y > z for x, y, z in zip(plan, pad, shape)]):
            raise ValueError("Chunk sizes %s plus padding sizes %s cannot exceed value dimensions %s along any axis"
                             % (tuple(plan), tuple(pad), tuple(shape)))

        if any([x > y for x, y in zip(pad, plan)]):
            raise ValueError("Padding sizes %s cannot exceed chunk sizes %s along any axis"
                             % (tuple(pad), tuple(plan)))

        def rectify(x):
            x[x<0] = 0
            return x

        breaks = [r_[arange(0, n, s), n] for n, s in zip(shape, plan)]
        limits = [zip(rectify(b[:-1]-p), b[1:]+p) for b, p in zip(breaks, pad)]
        slices = product(*[[slice(x[0], x[1]) for x in l] for l in limits])
        vals = [arr[s] for s in slices]
        newarr = empty(len(vals), dtype=object)
        for i in range(len(vals)):
            newarr[i] = vals[i]
        newsize = [b.shape[0]-1 for b in breaks]
        newarr = newarr.reshape(*newsize)
        return LocalChunks(newarr, shape, plan, dtype=arr.dtype, padding=pad)

    @staticmethod
    def getplan(size, shape, dtype, axes=None, padding=None):
        """
        Identify a plan for chunking values along each dimension.

        Generates an ndarray with the size (in number of elements) of chunks
        in each dimension. If provided, will estimate chunks for only a
        subset of axes, leaving all others to the full size of the axis.

        Parameters
        ----------
        size : string or tuple
             If str, the average size (in KB) of the chunks in all value dimensions.
             If int/tuple, an explicit specification of the number chunks in
             each moving value dimension.

        axes : tuple, optional, default=None
              One or more axes to estimate chunks for, if provided any
              other axes will use one chunk.

        padding : tuple or int, option, default=None
            Size over overlapping padding between chunks in each dimension.
            If tuple, specifies padding along each chunked dimension; if int,
            all dimensions use same padding; if None, no padding
        """
        from numpy import dtype as gettype

        # initialize with all elements in one chunk
        plan = asarray(shape)

        # check for subset of axes
        if axes is None:
            if isinstance(size, str):
                axes = arange(len(shape))
            else:
                axes = arange(len(size))
        else:
            axes = asarray(axes, 'int')

        # set padding
        pad = array(len(shape)*[0, ])
        if padding is not None:
            pad[axes] = padding

        # set the plan
        if isinstance(size, tuple):
            plan[axes] = size

        elif isinstance(size, str):
            # convert from kilobytes
            size = 1000.0 * float(size)

            # calculate from dtype
            elsize = gettype(dtype).itemsize
            nelements = prod(shape)
            dims = plan[axes]

            if size <= elsize:
                s = ones(len(axes))

            else:
                remsize = 1.0 * nelements * elsize
                s = []
                for (i, d) in enumerate(dims):
                    minsize = remsize/d
                    if minsize >= size:
                        s.append(1)
                        remsize = minsize
                        continue
                    else:
                        s.append(min(d, floor(size/minsize)))
                        s[i+1:] = plan[i+1:]
                        break

            plan[axes] = s

        else:
            raise ValueError("Chunk size not understood, must be tuple or int")

        return plan, pad
