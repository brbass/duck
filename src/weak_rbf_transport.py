import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys

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

class Cross_Section:
    def __init__(self,
                 val1,
                 val2):
        self.val1 = val1
        self.val2 = val2
    def val(self, x):
        if x < 1:
            return self.val1
        else:
            return self.val2

class Solution:
    def __init__(self,
                 sigma_t,
                 source,
                 psi0):
        self.sigma_t = sigma_t
        self.source = source
        self.psi0 = psi0
    
    def val(self,
            x):
        st1 = self.sigma_t.val1
        st2 = self.sigma_t.val2
        s1 = self.source.val1
        s2 = self.source.val2
        f0 = self.psi0
        
        if x < 1:
            return np.exp(-st1 * x) * ((np.exp(st1 * x) - 1) * s1 + f0 * st1) / st1
        else:
            return (s2 + (np.exp(-st1 + st2 - st2 * x) * (-s1 * st2 + f0 * st1 * st2 + np.exp(st1) * (-s2 * st1 + s1 * st2))) / st1) / st2
        
        
def solve_weak_transport(basis_str,
                         weight_str,
                         num_points,
                         ep_basis,
                         ep_weight,
                         sigma1,
                         sigma2,
                         source1,
                         source2,
                         psi0):
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    mu = 1
    
    # Set basis and weight functions
    if basis_str == "multiquadric":
        basis = Multiquadric(ep_basis,
                             points)
    elif basis_str == "gaussian":
        basis = Gaussian(ep_basis,
                         points)
    elif weight_str == "wendland":
        basis = Wendland(ep_basis,
                         points)
    else:
        print("basis not found: " + basis_str)
        return
    if weight_str == "multiquadric":
        weight = Multiquadric(ep_weight,
                              points)
    elif weight_str == "gaussian":
        weight = Gaussian(ep_weight,
                          points)
    elif weight_str == "wendland":
        weight = Wendland(ep_weight,
                          points)
    else:
        print("weight not found: " + weight_str)
        return

    # Set cross section and source
    sigma_t = Cross_Section(sigma1,
                            sigma2)
    source = Cross_Section(source1,
                           source2)
    
    # Initialize arrays
    a = np.zeros((num_points, num_points), dtype=float)
    b = np.zeros((num_points), dtype=float)

    # Set matrix
    for i in range(num_points):
        for j in range(num_points):
            def integrand(x):
                return (-mu * weight.dval(j, x) + sigma_t.val(x) * weight.val(j, x)) * basis.val(i, x)
            
            t1 = mu * basis.val(i, points[-1]) * weight.val(j, points[-1])
            t2, abserr = spi.quad(integrand, 0, length)
            a[j, i] = t1 + t2

    # Set RHS
    for j in range(num_points):
        def integrand(x):
            return weight.val(j, x) * source.val(x)

        t1 = mu * psi0 * weight.val(j, points[0])
        t2, abserr = spi.quad(integrand, 0, length, points=[1.])
        b[j] = t1 + t2
    
    alpha = spl.solve(a, b)

    psi = np.zeros(num_points)
    for i in range(num_points):
        val = 0.
        for j in range(num_points):
            val += alpha[j] * basis.val(j, points[i])
        psi[i] = val

    solution = Solution(sigma_t,
                        source,
                        psi0)
    analytic = np.zeros(num_points)
    for i in range(num_points):
        analytic[i] = solution.val(points[i])
    err = psi - analytic
    return err, psi
    
if __name__ == '__main__':
    if len(sys.argv) != 11:
        print("weak_rbf_transport [basis weight num_points ep_basis ep_weight sigma1 sigma2 source1 source2 psi0]")
        sys.exit()
    basis = str(sys.argv[1])
    weight = str(sys.argv[2])
    num_points = int(sys.argv[3])
    ep_basis = float(sys.argv[4])
    ep_weight = float(sys.argv[5])
    sigma1 = float(sys.argv[6])
    sigma2 = float(sys.argv[7])
    source1 = float(sys.argv[8])
    source2 = float(sys.argv[9])
    psi0 = float(sys.argv[10])
    err, psi = solve_weak_transport(basis,
                                    weight,
                                    num_points,
                                    ep_basis,
                                    ep_weight,
                                    sigma1,
                                    sigma2,
                                    source1,
                                    source2,
                                    psi0)
    print(err)
