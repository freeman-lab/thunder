"""
Utilities for representing and working with matrices as RDDs
"""

from numpy import dot, outer, shape, ndarray, mean, add, subtract, multiply, zeros, std, divide, sqrt
from pyspark.accumulators import AccumulatorParam

# TODO: right divide and left divide
# TODO: common operation is multiplying an RDD by its transpose times a matrix, how to do this cleanly?
# TODO: test using these in the various analyses packages (especially thunder.factorization)


class RowMatrix(object):
    """
    A large matrix backed by an RDD of (tuple, array) pairs
    The tuple can contain a row index, or any other useful
    identifier for each row.

    Parameters
    ----------
    rdd : RDD of (tuple, array) pairs
        RDD with matrix data

    nrows : int, optional, default = None
        Number of rows, will be automatially computed if not provided

    ncols : int, optional, default = None
        Number of columns, will be automatically computed if not provided

    """
    def __init__(self, rdd, nrows=None, ncols=None):
        self.rdd = rdd
        if nrows is None:
            self.nrows = rdd.count()
        else:
            self.nrows = nrows
        if ncols is None:
            vec = rdd.first()[1]
            if type(vec) is ndarray:
                self.ncols = len(vec)
            else:
                self.ncols = 1
        else:
            self.ncols = ncols

    def collect(self):
        """
        Collect the rows of the matrix, dropping the keys
        """
        return self.rdd.map(lambda (k, v): v).collect()

    def first(self):
        """
        Get the first row of the matrix
        """
        return self.rdd.first()[1]

    def rows(self):
        """
        Get the rows of the matrix, dropping the keys
        """
        return self.rdd.map(lambda (_, v): v)

    def cov(self, axis=None):
        """
        Compute a covariance matrix

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
        Compute gramian matrix (the product of the matrix with its transpose, i.e. A^T * A)

        Parameters
        ----------
        method : string, optional, default = "reduce"
            Method to use for summation
        """
        if method is "reduce":
            return self.rdd.map(lambda (k, v): v).mapPartitions(matrixsum_iterator_self).sum()

        if method is "accum":
            global mat
            mat = self.rdd.context.accumulator(zeros((self.ncols, self.ncols)), MatrixAccumulatorParam())

            def outerSum(x):
                global mat
                mat += outer(x, x)
            self.rdd.map(lambda (k, v): v).foreach(outerSum)

            return mat.value

        if method is "aggregate":

            def seqop(x, v):
                return x + outer(v, v)

            def combop(x, y):
                x += y
                return x

            return self.rdd.map(lambda (_, v): v).aggregate(zeros((self.ncols, self.ncols)), seqop, combop)

        else:
            raise Exception("method must be reduce or accum")

    def times(self, other, method="reduce"):
        """
        Multiply a RowMatrix by another matrix, either another RowMatrix
        or

        Parameters
        ----------
        other : RowMatrix, scalar, or numpy array
            Matrix to multiple with

        method : string, optional, default = "reduce"
            Method to use for summation
        """
        dtype = type(other)
        if dtype == RowMatrix:
            if self.nrows != other.nrows:
                raise Exception(
                    "cannot multiply shapes ("+str(self.nrows)+","+str(self.ncols)+") and ("+str(other.nrows)+","+str(other.ncols)+")")
            else:
                if method is "reduce":
                    return self.rdd.join(other.rdd).map(lambda (k, v): v).mapPartitions(matrixsum_iterator_other).sum()
                if method is "accum":
                    global mat
                    mat = self.rdd.context.accumulator(zeros((self.ncols, other.ncols)), MatrixAccumulatorParam())

                    def outersum(x):
                        global mat
                        mat += outer(x[0], x[1])
                    self.rdd.join(other.rdd).map(lambda (k, v): v).foreach(outersum)
                    return mat.value
                else:
                    raise Exception("method must be reduce or accum")
        else:
            if dtype == ndarray:
                dims = shape(other)
                if dims[0] != self.ncols:
                    raise Exception(
                        "cannot multiply shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
                if len(dims) == 0:
                    new_d = 1
                else:
                    new_d = dims[1]
            other_b = self.rdd.context.broadcast(other)
            return RowMatrix(self.rdd.mapValues(lambda x: dot(x, other_b.value)), self.nrows, new_d)

    def elementwise(self, other, op):
        """
        Apply an elementwise operation to a MatrixRDD

        Parameters
        ----------
        other : RowMatrix, scalar, or numpy array
            Matrix to multiple with

        op : function
            Binary operator to use for elementwise operations, e.g. add, subtract
        """
        dtype = type(other)
        if dtype is RowMatrix:
            if (self.nrows is not other.nrows) | (self.ncols is not other.ncols):
                raise Exception(
                    "cannot do elementwise op for shapes ("+self.nrows+","+self.ncols+") and ("+other.nrows+","+other.ncols+")")
            else:
                return RowMatrix(self.rdd.join(other.rdd).mapValues(lambda (x, y): op(x, y)), self.nrows, self.ncols)
        else:
            if dtype is ndarray:
                dims = shape(other)
                if len(dims) > 1 or dims[0] is not self.ncols:
                    raise Exception(
                        "cannot do elementwise operation for shapes ("+str(self.nrows)+","+str(self.ncols)+") and " + str(dims))
            return RowMatrix(self.rdd.mapValues(lambda x: op(x, other)), self.nrows, self.ncols)

    def plus(self, other):
        """
        Elementwise addition (see elementwise)
        """
        return RowMatrix.elementwise(self, other, add)

    def minus(self, other):
        """
        Elementwise subtraction (see elementwise)
        """
        return RowMatrix.elementwise(self, other, subtract)

    def dottimes(self, other):
        """
        Elementwise multiplcation (see elementwise)
        """
        return RowMatrix.elementwise(self, other, multiply)

    def dotdivide(self, other):
        """
        Elementwise division (see elementwise)
        """
        return RowMatrix.elementwise(self, other, divide)

    def sum(self):
        """
        Compute the row sum
        """
        return self.rdd.map(lambda (k, v): v).sum()

    def mean(self):
        """
        Compute the row mean
        """
        return self.rdd.map(lambda (k, v): v).sum() / self.nrows

    def var(self):
        """
        Compute the row sample variance
        """
        meanVec = self.mean()
        return self.rdd.map(lambda (k, v): (v - meanVec) ** 2).sum() / (self.nrows - 1)

    def std(self):
        """
        Compute the row standard deviation
        """
        return sqrt(self.var())

    def center(self, axis=0):
        """
        Center a RowMatrix in place by mean subtraction

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to center along, rows (0) or columns (1)
        """
        if axis is 0:
            self.rdd = self.rdd.mapValues(lambda x: x - mean(x))
            return self
        if axis is 1:
            meanvec = self.mean()
            self.rdd = self.minus(meanvec).rdd
        else:
            raise Exception("axis must be 0 or 1")

    def zscore(self, axis=0):
        """
        ZScore a RowMatrix in place by mean subtraction and division by standard deviation

        Parameters
        ----------
        axis : int, optional, default = 0
            Which axis to zscore along, rows (0) or columns (1)
        """
        if axis is 0:
            self.rdd = self.rdd.mapValues(lambda x: (x - mean(x))/std(x))
        if axis is 1:
            meanvec = self.mean()
            stdvec = self.std()
            self.rdd = self.minus(meanvec).dotdivide(stdvec).rdd
        else:
            raise Exception("axis must be 0 or 1")


def matrixsum_iterator_self(iterator):
    yield sum(outer(x, x) for x in iterator)


def matrixsum_iterator_other(iterator):
    yield sum(outer(x, y) for x, y in iterator)


class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return zeros(shape(value))

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1

