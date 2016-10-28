from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap

def get_offset(points,
               sigma_t,
               ep_weight):
    # pe = np.array([sigma_t.val(point) * 2 * (points[1] - points[0]) for point in points])
    # ga = np.cosh(pe) / np.sinh(pe) - 1. / pe
    # return 0.3 * ga * (points[1] - points[0])
    # return 0.5 * np.array([(points[1] - points[0]) / (sigma_t.val(point) + 0.5) for point in points])
    return 0.0 * np.array([ep_weight / (sigma_t.val(point) + 0.5) for point in points])
def weak_transport(basis_str,
                   weight_str,
                   num_points,
                   ep_basis,
                   ep_weight,
                   sigma1,
                   sigma2,
                   source1,
                   source2,
                   psi0,
                   plot_results = False):
    # Get problem description
    description = "weak_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(basis_str,
                                                                          weight_str,
                                                                          num_points,
                                                                          ep_basis,
                                                                          ep_weight,
                                                                          sigma1,
                                                                          sigma2,
                                                                          source1,
                                                                          source2,
                                                                          psi0)
    
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    mu = 1
    
    # Set cross section and source
    sigma_t = Cross_Section(sigma1,
                            sigma2)
    source = Cross_Section(source1,
                           source2)
    offset_distance = get_offset(points,
                                 sigma_t,
                                 ep_weight)
    
    # Set basis and weight functions
    if basis_str == "multiquadric":
        basis = Multiquadric(ep_basis,
                             points)
    elif basis_str == "gaussian":
        basis = Gaussian(ep_basis,
                         points)
    elif basis_str == "wendland":
        basis = Wendland(ep_basis,
                         points)
    elif basis_str == "mls":
        polyord = 2
        num_neighbors = 3
        basis = MLS(polyord,
                    num_neighbors,
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
    elif weight_str == "constant":
        weight = Constant(ep_weight,
                          points)
    elif weight_str == "compact_gaussian":
        weight = Compact_Gaussian(ep_weight,
                                  points)
    elif weight_str == "offset_gaussian":
        weight = Compact_Gaussian(ep_weight,
                                  points - offset_distance)
    elif weight_str == "offset_multiquadric":
        weight = Multiquadric(ep_weight,
                              points - offset_distance)
    elif weight_str == "mls":
        polyord = 2
        num_neighbors = 3
        weight = MLS(polyord,
                    num_neighbors,
                    points)
    else:
        print("weight not found: " + weight_str)
        return

    # Initialize arrays
    a = np.zeros((num_points, num_points), dtype=float)
    b = np.zeros((num_points), dtype=float)

    # Set matrix
    for i in range(num_points):
        for j in range(num_points):
            def integrand(x):
                return (-mu * weight.dval(j, x) + sigma_t.val(x) * weight.val(j, x)) * basis.val(i, x)
            limits = weight.limits(j)
            
            t1 = mu * basis.val(i, points[-1]) * weight.val(j, points[-1])
            t2, abserr = spi.quad(integrand, limits[0], limits[1])
            a[j, i] = t1 + t2
            
    # Set RHS
    for j in range(num_points):
        limits = weight.limits(j)
        def integrand(x):
            return weight.val(j, x) * source.val(x)

        t1 = mu * psi0 * weight.val(j, points[0])
        t2, abserr = spi.quad(integrand, limits[0], limits[1])
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
    l2err = np.divide(np.sqrt(np.sum(np.power(err, 2))), 1. * len(err))

    if True:
        description += "_l2err={:5e}".format(l2err)
        col = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ln1 = ax1.plot(points, analytic, label="analytic", color=col[0])
        ln2 = ax1.plot(points, psi, label="numeric", color=col[1])
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\psi(x)$")
        ax1.grid()
        ln3 = ax2.plot(points, err, label="error", color=col[2])
        ax2.set_ylabel(r"$err(\psi(x))$")
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        plt.title("\n".join(wrap(description, 60)))
        plt.savefig("../figs/{}.pdf".format(description))
        plt.close()
        
    return points, analytic, psi, err, l2err

if __name__ == '__main__':
    if len(sys.argv) != 11:
        print("weak_rbf_transport [basis weight num_points ep_basis ep_weight sigma1 sigma2 source1 source2 psi0]")
        sys.exit()
    i = itertools.count(1)
    basis = str(sys.argv[next(i)])
    weight = str(sys.argv[next(i)])
    num_points = int(sys.argv[next(i)])
    ep_basis = float(sys.argv[next(i)])
    ep_weight = float(sys.argv[next(i)])
    sigma1 = float(sys.argv[next(i)])
    sigma2 = float(sys.argv[next(i)])
    source1 = float(sys.argv[next(i)])
    source2 = float(sys.argv[next(i)])
    psi0 = float(sys.argv[next(i)])

    points, analytic, psi, err, l2err = weak_transport(basis,
                                                       weight,
                                                       num_points,
                                                       ep_basis,
                                                       ep_weight,
                                                       sigma1,
                                                       sigma2,
                                                       source1,
                                                       source2,
                                                       psi0)
