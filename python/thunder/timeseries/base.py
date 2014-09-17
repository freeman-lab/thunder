"""
Base class for calculating time series statistics
"""


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