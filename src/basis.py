import numpy as np
import sys
from matplotlib import pyplot as plt

# Basis and weight functions

class RBF:
    def __init__(self,
                 shape,
                 points):
        self.shape = shape / (points[1] - points[0])
        self.points = points
        self.compact = False
    def limits(self,
               i):
        return [self.points[0], self.points[-1]]

class Compact_RBF(RBF):
    def __init__(self,
                 shape,
                 points,
                 max_distance):
        RBF.__init__(self,
                     shape,
                     points)
        self.max_distance = max_distance
        self.compact = True
        self.limit = np.zeros((len(self.points), 2), dtype=float)
        for i, point in enumerate(self.points):
            self.limit[i, 0] = np.amax([0., self.points[i] - self.max_distance / self.shape])
            self.limit[i, 1] = np.amin([self.points[-1], self.points[i] + self.max_distance / self.shape])
            
    def limits(self,
               i):
        return self.limit[i, :]
        
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

class SUPG_Gaussian(Compact_RBF):
    def __init__(self,
                 shape,
                 points,
                 sigma_t):
        Compact_RBF.__init__(self,
                             shape,
                             points,
                             3.)
        self.compact_gaussian = Compact_Gaussian(shape,
                                                 points,
                                                 3.)
        self.sigma_t = sigma_t
        
    def val(self,
            i,
            x):
        return self.compact_gaussian.val(i,
                                         x) + self.compact_gaussian.dval(i,
                                                                         x) / self.sigma_t
    def dval(self,
             i,
             x):
        return self.compact_gaussian.dval(i,
                                          x) + self.compact_gaussian.ddval(i,
                                                                           x) / self.sigma_t
class Compact_Gaussian(Compact_RBF):
    def __init__(self,
                 shape,
                 points):
        Compact_RBF.__init__(self,
                             shape,
                             points,
                             3.)
    
    def val(self,
            i,
            x):
        d = x - self.points[i]
        if np.abs(d * self.shape) < self.max_distance:
            return np.exp(-np.power(d * self.shape, 2))
        else:
            return 0.
        
    def dval(self,
             i,
             x):
        d = x - self.points[i]
        if np.abs(d * self.shape) < self.max_distance:
            return -2 * np.power(self.shape, 2) * d * np.exp(-np.power(d * self.shape, 2))
        else:
            return 0.
    def ddval(self,
              i,
              x):
        d = x - self.points[i]
        if np.abs(d * self.shape) < self.max_distance:
            return 2. * np.exp(2) * np.power(self.shape, 2) * (2. * exp(2) * np.power(d, 2) - 1) * np.exp(-np.power(d * self.shape, 2))
        
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

class Wendland(Compact_RBF):
    def __init__(self,
                 shape,
                 points):
        Compact_RBF.__init__(self,
                             shape,
                             points,
                             1.)
        
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

class Constant(Compact_RBF):
    def __init__(self,
                 shape,
                 points):
        Compact_RBF.__init__(self,
                             shape,
                             points,
                             1.)
        
    def val(self,
            i,
            x):
        d = np.abs(x - self.points[i]) * self.shape
        if d < 1:
            return 1.
        else:
            return 0.
        
    def dval(self,
             i,
             x):
        d = np.abs(x - self.points[i]) * self.shape
        if d < 1:
            return 0.
        else:
            return 0.

if __name__ == '__main__':
    points = np.linspace(0, 1, 5)
    shape = 2.0
    funcs = [Wendland(shape,
                      points),
             Multiquadric(shape,
                          points),
             Gaussian(shape,
                      points),
             Constant(shape,
                      points),
             Compact_Gaussian(shape,
                              points)]
    desc = ["wend", "mult", "gauss", "const", "com_gauss"]
    col = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
    plot_points = np.linspace(0, 1, 200)
    plt.figure()
    plt.ylim(0, 2)
    for i, func in enumerate(funcs):
        for j, point in enumerate(points):
            vals = np.array([func.val(j, x) for x in plot_points])
            if j == 0:
                plt.plot(plot_points, vals, color=col[i], label=desc[i])
            else:
                plt.plot(plot_points, vals, color=col[i])
    plt.legend()
    plt.show()
