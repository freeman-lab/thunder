"""
Utilities for computing distances
"""

class Distance(object):

  def __init__(self, metric="euclidean"):
        self.metric = metric

  def get(self, d, q):

        if self.metric == "kl":
            l = len(d)
            g = abs(d * log(d / q))
            g[(d <= 1e-323) | (q <= 1e-323)] = 0
            return sum(g)

        if self.metric == "euclidean":
            return sum((d - q) ** 2)
