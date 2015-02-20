"""
Class with utilities for representing and working with matrices
"""
from numpy import dot, outer, shape, ndarray, add, subtract, multiply, zeros, divide, arange

from thunder.rdds.series import Series


# TODO: right divide and left divide
class RowMatrix(Series):
    """
    Distributed matrix data.

    Backed by an RDD of key-value pairs where the
    key is a tuple identifier, and the value is a one-dimensional array.

    Parameters
    ----------
    rdd : RDD of (tuple, array) pairs
        RDD containing the series data

    index : array-like or one-dimensional list
        Values must be unique, same length as the arrays in the input data.
        Defaults to arange(len(data)) if not provided.

    dims : Dimensions
        Specify the dimensions of the keys (min, max, and count),
        can avoid computation if known in advance

    nrows : int
        Number of rows, will be automatially computed if not provided,
        can save computation to specify if known in advance

    ncols : int
        Number of columns, will be automatically computed if not provided,
        can save computation to specify if known in advance

    """

    _metadata = Series._metadata + ['_ncols', '_nrows']

    def __init__(self, rdd, index=None, dims=None, dtype=None, nrows=None, ncols=None, nrecords=None,
                 awsCredentials=None):
        if nrows is not None and nrecords is not None:
            if nrows != nrecords:
                raise ValueError("nrows and nrecords must be consistent, got %d and %d" % (nrows, nrecords))
        nrecs = nrecords if nrecords is not None else nrows
        if index is not None:
            if ncols is not None and len(index) != ncols:
                    raise ValueError("ncols and len(index) must be consistent, got %d and len(index)=%d" %
                                     (ncols, len(index)))
        elif ncols is not None:
            index = arange(ncols)
        super(RowMatrix, self).__init__(rdd, nrecords=nrecs, dtype=dtype, dims=dims, index=index,
                                        awsCredentials=awsCredentials)

    @property
    def nrows(self):
        return self.nrecords

    @property
    def _nrows(self):
        # nrows is just an alias for nrecords
        return self._nrecords

    @property
    def ncols(self):
        return len(self.index)

    @property
    def _ncols(self):
        # ncols is an alias for len(self.index)
        return len(self._index) if self._index is not None else None

    @property
    def _constructor(self):
        return RowMatrix

    def rows(self):
        """
        Get the rows of the matrix, dropping the keys.
        """
        return self.rdd.map(lambda (_, v): v)

    def cov(self, axis=None):
        """
        Compute covariance of a distributed matrix.

        Parameters
        ----------
        axis : int, optional, default = None
            Axis for performing mean subtraction, None (no subtraction), 0 (rows) or 1 (columns)
        """
        if axis is None:
            return self.gramian() / self.nrows
        else:
            return self.center(axis).gramian() / self.nrows

    def gramian(self, method="accum"):
        """
        Compute gramian of a distributed matrix.

        The product of the matrix with its transpose, i.e. A^T * A

        Parameters
        ----------
        method : string, optional, default = "reduce"
            Method to use for summation
        """

        from pyspark.accumulators import AccumulatorParam

        class MatrixAccumulatorParam(AccumulatorParam):
            def zero(self, value):
                return zeros(shape(value))

            def addInPlace(self, val1, val2):
                val1 += val2
                return val1

        if method is "reduce":
            return self.rdd.map(lambda (k, v): v).mapPartitions(matrixSumIterator_self).sum()

        if method is "accum":
            global mat
            mat = self.rdd.context.accumulator(zeros((self.ncols, self.ncols)), MatrixAccumulatorParam())

            def outerSum(x):
                global mat
                mat += outer(x, x)
            self.rdd.map(lambda (k, v): v).foreach(outerSum)

            return mat.value

        if method is "aggregate":

            def seqOp(x, v):
                return x + outer(v, v)

            def combOp(x, y):
                x += y
                return x

            return self.rdd.map(lambda (_, v): v).aggregate(zeros((self.ncols, self.ncols)), seqOp, combOp)

        else:
            raise Exception("method must be reduce, accum, or aggregate")

    def times(self, other):
        """
        Multiply a RowMatrix by another matrix.

        Other matrix can be either another RowMatrix or a local matrix.
        NOTE: If multiplying two RowMatrices, they must have the same
        number of partitions and number of records per partition,
        e.g. because one was created through a map of the other,
        see zip

        Parameters
        ----------
        other : RowMatrix, scalar, or numpy array
            Matrix to multiply with

        method : string, optional, default = "reduce"
            Method to use for summation
        """
        dtype = type(other)
        if dtype == RowMatrix:
            if self.nrows != other.nrows:
                raise Exception(
                    "cannot multiply shapes (" + str(self.nrows) + "," + str(self.ncols) + ") and (" +
                    str(other.nrows) + "," + str(other.ncols) + ")")
            else:
                return self.rdd.zip(other.rdd).map(lambda ((k1, x), (k2, y)): (x, y))\
                    .mapPartitions(matrixSumIterator_other).sum()
        else:
            dims = shape(other)
            if dims[0] != self.ncols:
                raise Exception(
                    "cannot multiply shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
            if len(dims) == 0:
                new_d = 1
            else:
                new_d = dims[1]
            other_b = self.rdd.context.broadcast(other)
            newindex = arange(0, new_d)
            return self._constructor(self.rdd.mapValues(lambda x: dot(x, other_b.value)),
                                     nrows=self._nrows, ncols=new_d, index=newindex).__finalize__(self)

    def elementwise(self, other, op):
        """
        Apply an elementwise operation to distributed matrices.

        Can be applied to two RowMatrices,
        or between a RowMatrix and a local array.
        NOTE: For two RowMatrices, must have the same partitions
        and number of records per iteration (e.g. because
        one was created through a map on the other, see zip)

        Parameters
        ----------
        other : RowMatrix, scalar, or numpy array
            Matrix to combine with element-wise

        op : function
            Binary operator to use for elementwise operations, e.g. add, subtract
        """
        dtype = type(other)
        if dtype is RowMatrix:
            if (self.nrows is not other.nrows) | (self.ncols is not other.ncols):
                raise Exception(
                    "cannot do elementwise op for shapes ("+self.nrows+","+self.ncols+") and ("+other.nrows+","+other.ncols+")")
            else:
                return self._constructor(
                    self.rdd.zip(other.rdd).map(lambda ((k1, x), (k2, y)): (k1, add(x, y)))).__finalize__(self)
        else:
            if dtype is ndarray:
                dims = shape(other)
                if len(dims) > 1 or dims[0] is not self.ncols:
                    raise Exception(
                        "cannot do elementwise operation for shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
            return self._constructor(
                self.rdd.mapValues(lambda x: op(x, other))).__finalize__(self)

    def plus(self, other):
        """
        Elementwise addition of distributed matrices.

        See also
        --------
        elementwise

        """
        return RowMatrix.elementwise(self, other, add)

    def minus(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return RowMatrix.elementwise(self, other, subtract)

    def dotTimes(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return RowMatrix.elementwise(self, other, multiply)

    def dotDivide(self, other):
        """
        Elementwise division of distributed matrices.

        See also
        --------
        elementwise
        """
        return RowMatrix.elementwise(self, other, divide)


def matrixSumIterator_self(iterator):
    yield sum(outer(x, x) for x in iterator)


def matrixSumIterator_other(iterator):
    yield sum(outer(x, y) for x, y in iterator)




