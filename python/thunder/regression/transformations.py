from numpy import mean, std, hstack, ones

class Transform(object):
    """
    Class for transforming design matrix data before fitting/predicting.
    """

    def __init__(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class ZScore(Transform):
    """
    Class for rescaling data to units of standard deviations from the mean.
    """
    def __init__(self, X):
        self.mean = mean(X, axis=0)
        self.std = std(X, ddof=1, axis=0)

    def transform(self, X):
        return (X - self.mean)/self.std

class Center(Transform):
    """
    Class for centering data
    """
    def __init__(self, X):
        self.mean = mean(X, axis=0)

    def transform(self, X):
        return X - self.mean

class AddConstant(Transform):
    """
    Class for adding a column of 1s to a data matrix
    """
    def __init__(self):
        pass

    def transform(self, X):
        return hstack([ones([X.shape[0], 1]), X])


class TransformList(object):
    """
    Class for holding/applying a sequence of Transforms
    """
    def __init__(self, transforms=None):
        self.transforms = transforms

    def insert(self, transform):
        if self.transforms is None:
            self.transforms = []
        self.transforms.insert(0, transform)

    def apply(self, X):
        if self.transforms is None:
            return X
        for t in self.transforms:
            X = t.transform(X)
        return X


