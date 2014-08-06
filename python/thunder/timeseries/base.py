"""
Base and derived classes for calculating time series statistics
"""

from numpy import array, sqrt, fix, pi, median, std, sum, mean, shape, zeros, roll, dot, angle, abs
from scipy.linalg import norm
from scipy.io import loadmat
from numpy.fft import fft


class TimeSeriesBase(object):
    """Base class for doing calculations on time series"""

    def get(self, y):
        pass

    def calc(self, data):
        """Calculate a quantity (e.g. a statistic) on each record
        using the get function defined through a subclass

        Parameters
        ----------
        data : RDD of (tuple, array) pairs
            The data

        Returns
        -------
        params : RDD of (tuple, float) pairs
            The calculated quantity
        """

        params = data.mapValues(lambda x: self.get(x))
        return params


class Fourier(TimeSeriesBase):
    """Class for computing fourier transform"""

    def __init__(self, freq):
        self.freq = freq

    def get(self, y):
        """Compute fourier amplitude (coherence) and phase"""

        y = y - mean(y)
        nframes = len(y)
        ft = fft(y)
        ft = ft[0:int(fix(nframes/2))]
        amp_ft = 2*abs(ft)/nframes
        amp = amp_ft[self.freq]
        amp_sum = sqrt(sum(amp_ft**2))
        co = amp / amp_sum
        ph = -(pi/2) - angle(ft[self.freq])
        if ph < 0:
            ph += pi * 2
        return array([co, ph])


class Stats(TimeSeriesBase):
    """Class for computing simple summary statistics"""

    def __init__(self, statistic):
        self.func = {
            'median': lambda x: median(x),
            'mean': lambda x: mean(x),
            'std': lambda x: std(x),
            'norm': lambda x: norm(x - mean(x)),
        }[statistic]

    def get(self, y):
        """Compute the statistic"""

        return self.func(y)


class CrossCorr(TimeSeriesBase):
    """Class for computing lagged cross correlations

    Parameters
    ----------
    x : str, or array
        Signal to cross-correlate with, can be an array
        or location of MAT file with name sigfile_X.mat
        containing variable X with signal

    Attributes
    ----------
    x : array
        Signal to cross-correlate with
    """

    def __init__(self, sigfile, lag):
        if type(sigfile) is str:
            x = loadmat(sigfile + "_X.mat")['X'][0]
        else:
            x = sigfile
        x = x - mean(x)
        x = x / norm(x)

        if lag is not 0:
            shifts = range(-lag, lag+1)
            d = len(x)
            m = len(shifts)
            x_shifted = zeros((m, d))
            for ix in range(0, len(shifts)):
                tmp = roll(x, shifts[ix])
                if shifts[ix] < 0:  # zero padding
                    tmp[(d+shifts[ix]):] = 0
                if shifts[ix] > 0:
                    tmp[:shifts[ix]] = 0
                x_shifted[ix, :] = tmp
            self.x = x_shifted
        else:
            self.x = x

    def get(self, y):
        """Compute cross correlation between y and x"""

        y = y - mean(y)
        n = norm(y)
        if n == 0:
            b = zeros((shape(self.x)[0],))
        else:
            y /= norm(y)
            b = dot(self.x, y)
        return b
