from numpy import arctan2, sqrt, pi, array, size
from matplotlib import colors, cm
from thunder.utils.load import isrdd


class Colorize(object):
    """ Class for turning numerical data into colors"""

    def __init__(self, totype='rgb', scale=1):
        self.totype = totype
        self.scale = scale

    def calc(self, data):

        if isrdd(data):
            self.checkargs(size(data.first()[1]))
            return data.mapValues(lambda x: self.get(x))
        else:
            self.checkargs(size(data[0]))
            return map(lambda line: self.get(line), data)

    def get(self, line):
        if (self.totype == 'rgb') or (self.totype == 'hsv'):
            return clip(abs(line) * self.scale, 0, 1)

        elif self.totype == 'polar':
            theta = ((arctan2(line[1], line[0]) + pi + pi) % (2*pi)) / (2 * pi)
            rho = sqrt(line[0]**2 + line[1]**2)
            return clip(colors.hsv_to_rgb(array([[[theta, 1, rho * self.scale]]]))[0][0], 0, 1)

        else:
            if size(line) > 1:
                line = line[0]
            return array(cm.get_cmap(self.totype, 256)(line * self.scale)[0:3])

    def checkargs(self, n):

        if (self.totype == 'rgb' or self.totype == 'hsv') and n < 3:
            raise Exception("must have 3 values per record for rgb or hsv")
        elif self.totype == 'polar' and n < 1:
            raise Exception("must have at least 2 values per record for polar")


def clip(vals, minval, maxval):
    """Clip values below and above minimum and maximum values"""

    vals[vals < minval] = minval
    vals[vals > maxval] = maxval
    return array(vals)

