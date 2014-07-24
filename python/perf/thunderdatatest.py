import abc
import os
from datetime import datetime
from numpy import arange, array, add, float16, random, outer, dot, zeros, real, transpose, diag, argsort, sqrt, inner
from scipy.linalg import sqrtm, inv, orth, eig
from scipy.io import savemat
from thunder.io import load
from thunder.timeseries import Stats, Fourier, CrossCorr
from thunder.regression import RegressionModel
from thunder.factorization import SVD
from thunder.clustering import KMeans


class ThunderDataTest(object):

    def __init__(self, sc):
        self.sc = sc

    @abc.abstractmethod
    def runtest(self, **args):
        return

    @staticmethod
    def initialize(testname, sc):
        return TESTS[testname](sc)

    def createinputdata(self, numrecords, numdims, numpartitions):
        rdd = self.sc.parallelize(map(lambda x: (1, array([x])), arange(0, numrecords)), numpartitions)
        self.rdd = rdd

    def loadinputdata(self, datafile, savefile=None):
        rdd = load(self.sc, datafile, preprocessmethod="dff-percentile")
        self.rdd = rdd
        self.datafile = datafile
        if savefile is not None:
            self.savefile = savefile
        self.modelfile = os.path.join(os.path.split(self.datafile)[0], 'stim')

    def run(self, numtrials, persistencetype):

        if persistencetype == "memory":
            self.rdd.cache()
            self.rdd.count()

        def timedtest(func):
            start = datetime.now()
            func()
            end = datetime.now()
            dt = end - start
            time = (dt.microseconds + (dt.seconds + dt.days * 24.0 * 3600.0) * 10.0**6.0) / 10.0**6.0
            return time

        results = map(lambda i: timedtest(self.runtest), range(0, numtrials))

        return results


class StatsTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)
        self.method = Stats("std")

    def runtest(self):
        vals = self.method.calc(self.rdd)
        vals.count()


class AverageTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        vec = self.rdd.map(lambda (_, v): v).mean()


class RegressTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        model = RegressionModel.load(os.path.join(self.modelfile, "linear"), "linear")
        betas, stats, resid = model.fit(self.rdd)
        stats.count()


class RegressWithSaveTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        model = RegressionModel.load(os.path.join(self.modelfile, "linear"), "linear")
        betas, stats, resid = model.fit(self.rdd)
        result = stats.map(lambda (_, v): float16(v)).collect()
        savemat(self.savefile + "tmp.mat", mdict={"tmp": result}, oned_as='column')


class CrossCorrTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        method = CrossCorr(sigfile=os.path.join(self.modelfile, "crosscorr"), lag=0)
        betas = method.calc(self.rdd)
        betas.count()


class FourierTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)
        self.method = Fourier(freq=5)

    def runtest(self):
        vals = self.method.calc(self.rdd)
        vals.count()


class LoadTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        self.rdd.count()


class SaveTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        result = self.rdd.map(lambda (_, v): float16(v[0])).collect()
        savemat(self.savefile + "tmp.mat", mdict={"tmp": result}, oned_as='column')


class KMeansTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        centers = KMeans(3, maxiter=5, tol=0).train(self.rdd)


class ICATest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        k = len(self.rdd.first()[1])
        c = 3
        n = 1000
        B = orth(random.randn(k, c))
        Bold = zeros((k, c))
        iterNum = 0
        errVec = zeros(20)
        while (iterNum < 5):
            iterNum += 1
            B = self.rdd.map(lambda (_, v): v).map(lambda x: outer(x, dot(x, B) ** 3)).reduce(lambda x, y: x + y) / n - 3 * B
            B = dot(B, real(sqrtm(inv(dot(transpose(B), B)))))
            minAbsCos = min(abs(diag(dot(transpose(B), Bold))))
            Bold = B
            errVec[iterNum-1] = (1 - minAbsCos)

        sigs = self.rdd.mapValues(lambda x: dot(B, x))


class PCADirectTest(ThunderDataTest):
 
    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)
 
    def runtest(self):
        svd = SVD(3, method="direct").calc(self.rdd)


class PCAIterativeTest(ThunderDataTest):

    def __init__(self, sc):
        ThunderDataTest.__init__(self, sc)

    def runtest(self):
        m = len(self.rdd.first()[1])
        k = 3
        n = 1000

        def outerprod(x):
            return outer(x, x)

        c = random.rand(k, m)
        iter = 0
        error = 100

        while (iter < 5):
            c_old = c
            c_inv = dot(transpose(c), inv(dot(c, transpose(c))))
            premult1 = self.rdd.context.broadcast(c_inv)
            xx = self.rdd.map(lambda (_, v): v).map(lambda x: outerprod(dot(x, premult1.value))).sum()
            xx_inv = inv(xx)
            premult2 = self.rdd.context.broadcast(dot(c_inv, xx_inv))
            c = self.rdd.map(lambda (_, v): v).map(lambda x: outer(x, dot(x, premult2.value))).sum()
            c = transpose(c)
            error = sum(sum((c - c_old) ** 2))
            iter += 1


TESTS = {
    'stats': StatsTest,
    'average': AverageTest,
    'regress': RegressTest,
    'regresswithsave': RegressWithSaveTest,
    'crosscorr': CrossCorr,
    'fourier': Fourier,
    'load': LoadTest,
    'save': SaveTest,
    'ica': ICATest,
    'pca-direct': PCADirectTest,
    'pca-iterative': PCAIterativeTest,
    'kmeans': KMeansTest
}
