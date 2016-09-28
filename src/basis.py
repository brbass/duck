import numpy as np

# Basis and weight functions

class RBF:
    def __init__(self,
                 shape,
                 points):
        self.shape = shape / (points[1] - points[0])
        self.points = points
        
class Multiquadric(RBF):
    def __init__(self,
                 shape,
                 points):
        RBF.__init__(self,
                     shape,
                     points)
    def val(self,
            i,
            x):
        d = x - self.points[i]
        return np.sqrt(1 + np.power(d * self.shape, 2))
    def dval(self,
             i,
             x):
        d = x - self.points[i]
        return np.power(self.shape, 2) * d / np.sqrt(1 + np.power(d * self.shape, 2))

class Gaussian(RBF):
    def __init__(self,
                 shape,
                 points):
        RBF.__init__(self,
                     shape,
                     points)
    def val(self,
            i,
            x):
        d = x - self.points[i]
        return np.exp(-np.power(d * self.shape, 2))
    def dval(self,
             i,
             x):
        d = x - self.points[i]
        return -2 * np.power(self.shape, 2) * d * np.exp(-np.power(d * self.shape, 2))

class Wendland(RBF):
    def __init__(self,
                 shape,
                 points):
        RBF.__init__(self,
                     shape,
                     points)
    def val(self,
            i,
            x):
        d = np.abs(x - self.points[i]) * self.shape
        if d < 1:
            return np.power(1 - d, 4) * (1 + 4 * d)
        else:
            return 0.
        
    def dval(self,
             i,
             x):
        d = np.abs(x - self.points[i]) * self.shape
        if d < 1:
            return 20 * self.shape * d * np.power(d - 1, 3)
        else:
            return 0.

