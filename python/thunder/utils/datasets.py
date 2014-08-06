"""
Utilities for loading test datasets
"""

import os
from numpy import array, random, shape, floor, dot, linspace, sin, sign, c_
from thunder.utils import load


class DataSets(object):

    def __init__(self, sc):
        self.sc = sc
        self.path = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
    def make(sc, name, **opts):
        return DATASET_MAKERS[name](sc).generate(**opts)

    @staticmethod
    def load(sc, name):
        return DATASET_LOADERS[name](sc).fromfile()


def appendkeys(data):

    data = array(data)
    n = shape(data)[0]
    x = (random.rand(n) * n).astype(int)
    y = (random.rand(n) * n).astype(int)
    z = (random.rand(n) * n).astype(int)
    data_zipped = zip(x, y, z, data)
    return map(lambda (k1, k2, k3, v): ((k1, k2, k3), v), data_zipped)


class KMeansData(DataSets):

    def generate(self, k=5, npartitions=10, ndims=5, nrecords=100, noise=0.1, seed=None, params=False):
        random.seed(seed)
        centers = random.randn(k, ndims)
        gen_func = lambda i: centers[int(floor(random.rand(1, 1) * k))] + noise*random.rand(ndims)
        data_local = map(gen_func, range(0, nrecords))
        data = self.sc.parallelize(appendkeys(data_local), npartitions)
        if params is True:
            return data, centers
        else:
            return data


class PCAData(DataSets):

    def generate(self, k=3, npartitions=10, nrows=100, ncols=10, seed=None, params=False):
        random.seed(seed)
        u = random.randn(nrows, k)
        v = random.randn(k, ncols)
        a = dot(u, v)
        a += random.randn(shape(a)[0], shape(a)[1])
        data = self.sc.parallelize(appendkeys(a), npartitions)
        if params is True:
            return data, u, v
        else:
            return data


class ICAData(DataSets):

    def generate(self, npartitions=10, nrows=100, params=False):
        random.seed(42)
        time = linspace(0, 10, nrows)
        s1 = sin(2 * time)
        s2 = sign(sin(3 * time))
        s = c_[s1, s2]
        s += 0.2 * random.randn(s.shape[0], s.shape[1])  # Add noise
        s /= s.std(axis=0)
        a = array([[1, 1], [0.5, 2]])
        x = dot(s, a.T)
        data = self.sc.parallelize(appendkeys(x), npartitions)
        if params is True:
            return data, s, a
        else:
            return data


class FishData(DataSets):

    def fromfile(self):
        return load(self.sc, os.path.join(self.path, 'data/fish.txt'))


class IrisData(DataSets):

    def fromfile(self):
        return load(self.sc, os.path.join(self.path, 'data/iris.txt'))


DATASET_MAKERS = {
    'kmeans': KMeansData,
    'pca': PCAData,
    'ica': ICAData
}

DATASET_LOADERS = {
    'iris': IrisData,
    'fish': FishData
}