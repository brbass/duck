from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys

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
