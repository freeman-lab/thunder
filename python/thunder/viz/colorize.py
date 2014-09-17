from numpy import arctan2, sqrt, pi, array, size, shape, ones, abs, dstack, clip, transpose, zeros
import colorsys
from matplotlib import colors, cm


class Colorize(object):
    """Class for turning numerical data into colors
    Can operate over either points or images
    
    Parameters
    ----------
    totype : string, optional, default = Pastel1
        The color to convert to

    scale : float, optional, default = 1
        How to scale amplitude during color conversion, controls brighthness
    """

    def __init__(self, totype='Pastel1', scale=1):
        self.totype = totype
        self.scale = scale

    def points(self, pts):
        """Colorize a set of points or a single points. Input must have
        either one dimension (a single point) or two dimensions (a list or
        array of points). 

        Parameters
        ----------
        pts : array
            The point or points to colorize. Must be of shape (n, c) or (c,) where
            c is the dimension containing the information for colorizing. 

        Returns
        -------
        out : array
            Color assignments for each point, either (n, 3) or (3,)
        """
        pts = array(pts)
        n = len(pts[0])
        self.checkargs(n)

        if pts.ndim > 2:
            raise Exception("points must have 1 or 2 dimensions")

        if pts.ndim == 1:
            out = clip(self.get(pts), 0, 1)
        else:
            out = map(lambda line: clip(self.get(line), 0, 1), pts)

        return out

    def image(self, img):
        """Colorize an image. Input can either be a single image or a stack of images.
        In either case, the first dimension must be the quantity to be used for colorizing.

        Parameters
        ----------
        img : array
            The image to colorize. Must be of shape (c, x, y, z) or (c, x, y), where
            c is the dimension containing the information for colorizing.

        Returns
        -------
        out : array
            Color assignments for images, either (x, y, z, 3) or (x, y, 3)
        """

        d = shape(img)
        self.checkargs(d[0])

        if img.ndim > 4 or img.ndim < 3:
            raise Exception("image data must have 3 or 4 dimensions, first is for coloring, remainder are xy(z)")

        if (self.totype == 'rgb') or (self.totype == 'hsv'):
            out = abs(img) * self.scale
            if img.ndim == 4:
                out = transpose(out, (1, 2, 3, 0))
            if img.ndim == 3:
                out = transpose(out, (1, 2, 0))

        elif self.totype == 'polar':
            theta = ((arctan2(-img[0], -img[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(img[0]**2 + img[1]**2)
            if img.ndim == 4:
                saturation = ones((d[1],d[2]))
                out = zeros((d[1], d[2], d[3], 3))
                for i in range(0, d[3]):
                    out[:, :, i, :] = colors.hsv_to_rgb(dstack((theta[:, :, i], saturation, self.scale*rho[:, :, i])))
            if img.ndim == 3:
                saturation = ones((d[1], d[2]))
                out = colors.hsv_to_rgb(dstack((theta, saturation, self.scale*rho)))

        else:
            out = cm.get_cmap(self.totype, 256)(img[0] * self.scale)
            if img.ndim == 4:
                out = out[:, :, :, 0:3]
            if img.ndim == 3:
                out = out[:, :, 0:3]

        return clip(out, 0, 1)

    def get(self, line):
        
        if (self.totype == 'rgb') or (self.totype == 'hsv'):
            return abs(line) * self.scale

        if self.totype == 'polar':
            theta = ((arctan2(-line[0], -line[1]) + pi/2) % (pi*2)) / (2 * pi)
            rho = sqrt(line[0]**2 + line[1]**2)
            return colorsys.hsv_to_rgb(theta, 1, rho * self.scale)

        else:
            return cm.get_cmap(self.totype, 256)(line[0] * self.scale)[0:3]

    def checkargs(self, n):

        if (self.totype == 'rgb' or self.totype == 'hsv') and n < 3:
            raise Exception("must have 3 values per record for rgb or hsv")
        elif self.totype == 'polar' and n < 1:
            raise Exception("must have at least 2 values per record for polar")
