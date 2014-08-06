"""
Utilities for loading and preprocessing data
"""

import os
import glob
import pyspark
from numpy import array, mean, cumprod, append, mod, ceil, size, polyfit, polyval, arange, percentile, inf, subtract
from scipy.signal import butter, lfilter


class Dimensions(object):
    """Helper class for estimating and storing dimensions of data
    based on the keys"""

    def __init__(self, values=[], n=3):
        self.min = tuple(map(lambda i: inf, range(0, n)))
        self.max = tuple(map(lambda i: -inf, range(0, n)))

        for v in values:
            self.merge(v)

    def merge(self, value):
        self.min = tuple(map(min, self.min, value))
        self.max = tuple(map(max, self.max, value))
        return self

    def count(self):
        return tuple(map(lambda x: x + 1, map(subtract, self.max, self.min)))

    def mergedims(self, other):
        self.min = tuple(map(min, self.min, other.min))
        self.max = tuple(map(max, self.max, other.max))
        return self


class Parser(object):
    """Class for parsing lines of a data file"""

    def __init__(self, nkeys):
        def func(line):
            vec = [float(x) for x in line.split(' ')]
            ts = array(vec[nkeys:])
            keys = tuple(int(x) for x in vec[:nkeys])
            return keys, ts

        self.func = func

    def get(self, y):
        return self.func(y)


class PreProcessor(object):
    """Class for preprocessing data"""

    def __init__(self, preprocessmethod):
        if preprocessmethod == "sub":
            func = lambda y: y - mean(y)

        if preprocessmethod == "dff":
            def func(y):
                mnval = mean(y)
                return (y - mnval) / (mnval + 0.1)

        if preprocessmethod == "raw":
            func = lambda x: x

        if preprocessmethod == "dff-percentile":

            def func(y):
                mnval = percentile(y, 20)
                y = (y - mnval) / (mnval + 0.1)   
                return y

        if preprocessmethod == "dff-detrend":

            def func(y):
                mnval = mean(y)
                y = (y - mnval) / (mnval + 0.1)   
                x = arange(1, len(y)+1) 
                p = polyfit(x, y, 1)
                yy = polyval(p, x)
                return y - yy

        if preprocessmethod == "dff-detrendnonlin":

            def func(y):
                mnval = mean(y)
                y = (y - mnval) / (mnval + 0.1)   
                x = arange(1, len(y)+1) 
                p = polyfit(x, y, 5)
                yy = polyval(p, x)
                return y - yy

        if preprocessmethod == "dff-detrend-percentile":

            def func(y):
                # first do linear detrending (but don't subtract the intercept)
                x = arange(1, len(y)+1)
                p = polyfit(x, y, 1)
                yy = p[0] * x  # subtract off just the slope term
                y = y - yy

                # then take 20th percentile as baseline
                mnval = percentile(y, 20)
                y = (y - mnval) / (mnval + 0.1)
                return y

        if preprocessmethod == "dff-highpass":
            fs = 1
            nyq = 0.5 * fs
            cutoff = (1.0/360) / nyq
            b, a = butter(6, cutoff, "highpass")

            def func(y):
                mnval = mean(y)
                y = (y - mnval) / (mnval + 0.1)
                return lfilter(b, a, y)

        self.func = func

    def get(self, y):
        return self.func(y)


def isrdd(data):
    """ Check whether data is an RDD or not"""

    dtype = type(data)
    if (dtype == pyspark.rdd.RDD) | (dtype == pyspark.rdd.PipelinedRDD):
        return True
    else:
        return False


def getdims(data):
    """Get dimensions of data via the keys. Ranges can have arbtirary minima
    and maximum, but they must be contiguous (e.g. the indices of a dense matrix)."""

    def redfunc(left, right):
        return left.mergedims(right)

    if isrdd(data):
        entry = data.first()[0]
        n = size(entry)
        d = data.map(lambda (k, _): k).mapPartitions(lambda i: [Dimensions(i, n)]).reduce(redfunc)
    else:
        entry = data[0][0]
        rng = range(0, size(entry))
        d = Dimensions()
        if size(entry) == 1:
            distinctvals = list(set(map(lambda x: x[0][0], data)))
        else:
            distinctvals = map(lambda i: list(set(map(lambda x: x[0][i], data))), rng)
        d.max = tuple(map(max, distinctvals))
        d.min = tuple(map(min, distinctvals))

    return d


def subtoind(data, dims):
    """Convert subscript indexing to linear indexing"""

    def subtoind_inline(k, dimprod):
        return sum(map(lambda (x, y): (x - 1) * y, zip(k[1:], dimprod))) + k[0]
    if size(dims) > 1:
        dimprod = cumprod(dims)[0:-1]
        if isrdd(data):
            return data.map(lambda (k, v): (subtoind_inline(k, dimprod), v))
        else:
            return map(lambda (k, v): (subtoind_inline(k, dimprod), v), data)
    else:
        return data.map(lambda (k, v): (k[0], v))


def indtosub(data, dims):
    """Convert linear indexing to subscript indexing"""

    def indtosub_inline(k, dimprod):
        return tuple(map(lambda (x, y): int(mod(ceil(float(k)/y) - 1, x) + 1), dimprod))

    if size(dims) > 1:
        dimprod = zip(dims, append(1, cumprod(dims)[0:-1]))
        if isrdd(data):
            return data.map(lambda (k, v): (indtosub_inline(k, dimprod), v))
        else:
            return map(lambda (k, v): (indtosub_inline(k, dimprod), v), data)

    else:
        return data


def load(sc, datafile, preprocessmethod="raw", nkeys=3, npartitions=None):
    """Load data from a text file (or a directory of files) with format
    <k1> <k2> ... <t1> <t2> ...
    where <k1> <k2> ... are keys (Int) and <t1> <t2> ... are the data values (Double)
    If passed a directory, will automatically sort files by name

    Parameters
    ----------
    sc : SparkContext
        The Spark Context

    datafile : str
        Location of raw data

    preprocessmethod : str, optional, default = "raw" (no preprocessing)
        Which preprocessing to perform

    nkeys : int, optional, default = 3
        Number of keys per data point

    npartitions : int, optional, default = None
        Number of partitions

    Returns
    -------
    data : RDD of (tuple, array) pairs
        The parsed and preprocessed data as an RDD
    """
    # TODO: Add support for binary files using the new hadoop input format support

    if os.path.isdir(datafile):
        files = sorted(glob.glob(os.path.join(datafile, '*')))
        datafile = ''.join([files[x] + ',' for x in range(0, len(files))])[0:-1]

    lines = sc.textFile(datafile, npartitions)
    parser = Parser(nkeys)

    data = lines.map(parser.get)

    if preprocessmethod != "raw":
        preprocessor = PreProcessor(preprocessmethod)
        data = data.mapValues(preprocessor.get)

    return data


