"""
Class and standalone app for mass-univariate classification
"""

import argparse
from numpy import in1d, zeros, array, size, float64
from scipy.io import loadmat
from scipy.stats import ttest_ind
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from thunder.utils import ThunderContext, save


class MassUnivariateClassifier(object):
    """Base class for mass univariate classification.
    Assumes that for each signal, each time point belongs to one
    of several categories. Uses independent classifiers to train and predict.
    Example usage: determining how well each individual neural signal
    predicts a behavior.

    Parameters
    ----------
    paramfile : str or dict
        A MAT file or Python dictionary containing parameters for training classifiers.
        At minimum must contain a "labels" field, with the label the classify at each
        time point. Can additionally include fields for "features" (which feature
        was present at each time point) and "samples" (which sample was present
        at each time point)

    Attributes
    ----------
    `labels` : array
        Array containing the label for each time point

    `features` : array
        Which feature was present at each time point

    `nfeatures` : int, optional
        Number of features

    `samples` : array
        Which sample does each time point belong to

    `sampleids` : list
        Unique samples

    `nsamples` : int
        Number of samples
    """

    def __init__(self, paramfile):
        if type(paramfile) is str:
            params = loadmat(paramfile, squeeze_me=True)
        elif type(paramfile) is dict:
            params = paramfile
        else:
            raise TypeError("Parameters for classification must be provided as string with file location, or dictionary")

        self.labels = params['labels']

        if 'features' in params:
            self.features = params['features']
            self.nfeatures = len(list(set(self.features.flatten())))
            self.samples = params['samples']
            self.sampleids = list(set(self.samples.flatten()))
            self.nsamples = len(self.sampleids)
        else:
            self.nfeatures = 1
            self.nsamples = len(self.labels)

    @staticmethod
    def load(paramfile, classifymode, **opts):
        return CLASSIFIERS[classifymode](paramfile, **opts)

    def get(self, x, set=None):
        pass

    def classify(self, data, featureset=None):
        """Run classification on an RDD

        Parameters
        ----------
        data: RDD of (tuple, array) pairs
            The data

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        perf : RDD of scalars
            The performance of the classifer for each record
        """

        if self.nfeatures == 1:
            perf = data.mapValues(lambda x: [self.get(x)])
        else:
            if featureset is None:
                featureset = [[self.features[0]]]
            for i in featureset:
                assert array([item in i for item in self.features]).sum() != 0, "Feature set invalid"
            perf = data.mapValues(lambda x: map(lambda i: self.get(x, i), featureset))

        return perf


class GaussNaiveBayesClassifier(MassUnivariateClassifier):
    """Classifier for Gaussian Naive Bayes classification

    Parameters
    ----------
    paramfile : str or dict
        Parameters for classification, see MassUnivariateClassifier

    cv : int
        Folds of cross-validation, none if 0
    """

    def __init__(self, paramfile, cv=0):
        MassUnivariateClassifier.__init__(self, paramfile)

        self.cv = cv
        self.func = GaussianNB()

    def get(self, x, featureset=None):
        """Compute classification performance

        Parameters
        ----------
        x : array
            Data for a single record

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        perf : scalar
            Performance of the classifier on this record
        """

        y = self.labels
        if self.nfeatures == 1:
            X = zeros((self.nsamples, 1))
            X[:, 0] = x
        else:
            X = zeros((self.nsamples, size(featureset)))
            for i in range(0, self.nsamples):
                inds = (self.samples == self.sampleids[i]) & (in1d(self.features, featureset))
                X[i, :] = x[inds]

        if self.cv > 0:
            return cross_validation.cross_val_score(self.func, X, y, cv=self.cv).mean()
        else:
            ypred = self.func.fit(X, y).predict(X)
            perf = array(y == ypred).mean()
            return perf


class TTestClassifier(MassUnivariateClassifier):
    """Classifier for TTest classification

    Parameters
    ----------
    paramfile : str or dict
     Parameters for classification, see MassUnivariateClassifier
    """

    def __init__(self, paramfile):
        MassUnivariateClassifier.__init__(self, paramfile)

        self.func = ttest_ind
        unique = list(set(list(self.labels)))
        if len(unique) != 2:
            raise TypeError("Only two types of labels allowed for t-test classificaiton")
        if unique != set((0, 1)):
            self.labels = array(map(lambda i: 0 if i == unique[0] else 1, self.labels))

    def get(self, x, featureset=None):
        """Compute classification performance as a t-statistic

        Parameters
        ----------
        x : array
            Data for a single record

        featureset : array, optional, default = None
            Which features to use

        Returns
        -------
        t : scalar
            t-statistic for this record
        """

        if (self.nfeatures > 1) & (size(featureset) > 1):
            X = zeros((self.nsamples, size(featureset)))
            for i in range(0, size(featureset)):
                X[:, i] = x[self.features == featureset[i]]
            t = float64(self.func(X[self.labels == 0, :], X[self.labels == 1, :])[0])

        else:
            if self.nfeatures > 1:
                x = x[self.features == featureset]
            t = float64(self.func(x[self.labels == 0], x[self.labels == 1])[0])

        return t


CLASSIFIERS = {
    'gaussnaivebayes': GaussNaiveBayesClassifier,
    'ttest': TTestClassifier
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fit a regression model")
    parser.add_argument("datafile", type=str)
    parser.add_argument("paramfile", type=str)
    parser.add_argument("outputdir", type=str)
    parser.add_argument("classifymode", choices="naivebayes", help="form of classifier")
    parser.add_argument("--featureset", type=array, default="None", required=False)
    parser.add_argument("--cv", type=int, default="0", required=False)
    parser.add_argument("--preprocess", choices=("raw", "dff", "sub", "dff-highpass", "dff-percentile"
                        "dff-detrendnonlin", "dff-detrend-percentile"), default="raw", required=False)

    args = parser.parse_args()

    tsc = ThunderContext.start("classify")

    data = tsc.loadText(args.datafile, args.preprocess)
    clf = MassUnivariateClassifier.load(args.paramfile, args.classifymode, cv=args.cv)
    perf = clf.classify(data, args.featureset)

    outputdir = args.outputdir + "-classify"
    save(perf, outputdir, "perf", "matlab")
