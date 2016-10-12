import numpy as np
import sys
from matplotlib import pyplot as plt
from two_region import Cross_Section

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
                                                 points)
        self.delta_sigma_t = 0.1
        self.sigma_t = sigma_t
        
    def val(self,
            i,
            x):
        return self.compact_gaussian.val(i,
                                         x) + self.compact_gaussian.dval(i,
                                                                         x) / (self.sigma_t.val(self.points[i]) + self.delta_sigma_t)
    def dval(self,
             i,
             x):
        return self.compact_gaussian.dval(i,
                                          x) + self.compact_gaussian.ddval(i,
                                                                           x) / (self.sigma_t.val(self.points[i]) + self.delta_sigma_t)

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
            return 2. * np.exp(2) * np.power(self.shape, 2) * (2. * np.exp(2) * np.power(d, 2) - 1) * np.exp(-np.power(d * self.shape, 2))
        else:
            return 0.
        
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

class MLS:
    def __init__(self,
                 num_polynomials,
                 num_other_points,
                 points):
        self.num_polynomials = num_polynomials
        self.num_other_points = num_other_points
        self.points = points
        self.num_points = len(points)
        self.dx = points[1] - points[0]
        self.bandwidth = 1. / (self.dx * (self.num_other_points + 0.1))
    def weight(d):
        if d <= 1:
            return np.power(1. - d, 3) * (1. + 3. * d)
        else:
            return 0.
    def dweight(d):
        if d <= 1:
            return -12 * self.bandwidth * np.power(d - 1, 2) * d
        else:
            return 0.
    def weight(i,
               x):
        return self.weight(self.bandwidth * np.abs(x - self.points[i]))
    def dweight(i,
                x):
        return self.dweight(self.bandwidth * np.abs(x - self.points[i]))
    def get_points(x):
        nearest_point = np.round(x / self.dx)
        local_points = []
        for i in range(nearest_point - self.num_other_points - 1, nearest_point + self.num_other_points + 1):
            if (i >= 0 and i < self.num_points
                and np.abs(x - self.points[i]) < bandwidth):
                local_points.append([i])
        return np.array(local_points, dtype=int)
    def amat(x):
        local_points = self.get_points(x)
        mat = np.zeros((self.num_polynomials, self.num_polynomials))
        
        for i in local_points:
            w = weight(i,
                       x)
            for j in range(self.num_polynomials):
                for k in range(self.num_polynomials):
                    mat[j, k] = mat[j, k] + np.power(self.points[i], j + k) * w
        return mat
    def damat(x):
        local_points = self.get_points(x)
        mat = np.zeros((self.num_polynomials, self.num_polynomials))
        
        for i in local_points:
            dw = dweight(i,
                         x)
            for j in range(self.num_polynomials):
                for k in range(self.num_polynomials):
                    mat[j, k] = mat[j, k] + np.power(self.points[i], j + k) * dw
        return mat
    def ainv(x):
        return np.linalg.inverse(amat(x))
    def dainv(x):
        ainvval = ainv(x)
        daval = damat(x)
        return -1. * np.dot(ainvval, np.dot(daval, ainvval))
    def poly(x):
        vec = np.zeros(self.num_polynomials)
        for i in range(self.num_polynomials):
            vec[i] = np.power(x, i)
        return vec
    def dpoly(x):
        vec = np.zeros(self.num_polynomials)
        for i in range(1, self.num_polynomials):
            vec[i] = i * np.power(x, i - 1)
        return vec
    def bvec(i,
             x):
        return poly(points[i]) * weight(i,
                                        x)
    def dbvec(i,
              x):
        return poly(points[i]) * dweight(i,
                                         x)
        
    def val(self,
            i,
            x):
        return np.dot(poly(x), np.dot(ainv(x), bvec(i,
                                                    x)))
    def dval(self,
             i,
             x):
        polyval = poly(x)
        dpolyval = dpoly(x)
        aval = amat(x)
        daval = damat(x)
        ainvval = np.linalg.inverse(amat(x))
        bval = bvec(i,
                    x)
        dbval = dbvec(i,
                      x)
        dainvval = -np.dot(ainvval, np.dot(daval, ainvval))
        t1 = np.dot(dpolyval, np.dot(ainvval, bval))
        t2 = np.dot(polyval, dainvval, bval)
        t3 = np.dot(polyval, ainvval, dbval)
        return t1 + t2 + t3

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
    points = np.linspace(0, 1, 3)
    shape = 2.0
    sigma_t = Cross_Section(1.0,
                            2.0)
    funcs = [Wendland(shape,
                      points),
             Multiquadric(shape,
                          points),
             Gaussian(shape,
                      points),
             Constant(shape,
                      points),
             Compact_Gaussian(shape,
                              points),
             SUPG_Gaussian(shape,
                           points,
                           sigma_t)]
    desc = ["wend", "mult", "gauss", "const", "com_gauss", "supg_gauss"]
    col = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854', 'k']
    plot_points = np.linspace(0, 1, 200)
    plt.ylim(0, 2)
    for i, func in enumerate(funcs):
        for j, point in enumerate(points):
            vals = np.array([func.val(j, x) for x in plot_points])
            dvals = np.array([func.dval(j, x) for x in plot_points])
            if j == 0:
                plt.figure(0)
                plt.plot(plot_points, vals, color=col[i], label=desc[i])
                plt.figure(1)
                plt.plot(plot_points, dvals, color=col[i], label=desc[i])
            else:
                plt.figure(0)
                plt.plot(plot_points, vals, color=col[i])
                plt.figure(1)
                plt.plot(plot_points, dvals, color=col[i])
    plt.figure(0)
    plt.legend()
    plt.figure(1)
    plt.legend()
    plt.show()
