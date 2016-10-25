from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap
from enum import Enum

def get_offset(points,
               sigma_t,
               ep_weight):
    # pe = np.array([sigma_t.val(point) * 2 * (points[1] - points[0]) for point in points])
    # ga = np.cosh(pe) / np.sinh(pe) - 1. / pe
    # return 0.3 * ga * (points[1] - points[0])
    # return 0.5 * np.array([(points[1] - points[0]) / (sigma_t.val(point) + 0.5) for point in points])
    return 0.0 * np.array([ep_weight / (sigma_t.val(point) + 0.5) for point in points])

class CS_Method(Enum):
    full = 0
    flux = 1
    weight = 2
    point = 3

# Vector support for scalar function
def vfunc(xvals, func):
    for i, x in enumerate(xvals):
        xvals[i] = func(x)
    return xvals
# def vfunc(xvals, func):
#     res = np.empty(len(xvals))
#     for i, x in enumerate(xvals):
#         res[i] = func(x)
#     return res
    
def supg_transport(basis_str,
                   weight_str,
                   cs_method,
                   num_points,
                   ep_basis,
                   ep_weight,
                   tau1,
                   tau2,
                   sigma1,
                   sigma2,
                   source1,
                   source2,
                   psi0,
                   fixed_quadrature = True):
    # Integration order
    int_ord = 16
    
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    mu = 1
    
    # Set cross section and source
    sigma_t = Cross_Section(sigma1,
                            sigma2)
    tau = Cross_Section(tau1,
                        tau2)
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
    elif weight_str == "mls":
        polyord = 2
        num_neighbors = 3
        weight = MLS(polyord,
                    num_neighbors,
                    points)
    else:
        print("weight not found: " + weight_str)
        return

    # Set problem solution
    solution = Solution(sigma_t,
                        source,
                        psi0)
    
    # Calculate sigma_t (if applicable)
    sigma_t_vals = np.zeros(num_points)
    if cs_method is CS_Method.full:
        pass
    elif cs_method is CS_Method.flux:
        for i in range(num_points):
            limits = weight.limits(i)
            def integrand1(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                ta = tau.val(x)
                st = sigma_t.val(x)
                sol = solution.val(x)
                return (wei + ta * mu * dwei) * st * sol
            def integrand2(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                ta = tau.val(x)
                sol = solution.val(x)
                return (wei + ta * mu * dwei) * sol
            if fixed_quadrature:
                int1, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand1], n=int_ord)
                int2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand2], n=int_ord)
            else:
                int1, err = spi.quad(integrand1, limits[0], limits[1])
                int2, err = spi.quad(integrand2, limits[0], limits[1])
            sigma_t_vals[i] = int1 / int2
    elif cs_method is CS_Method.weight:
        for i in range(num_points):
            limits = weight.limits(i)
            def integrand1(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                ta = tau.val(x)
                st = sigma_t.val(x)
                return (wei + ta * mu * dwei) * st
            def integrand2(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                ta = tau.val(x)
                return wei + ta * mu * dwei
            if fixed_quadrature:
                int1, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand1], n=int_ord)
                int2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand2], n=int_ord)
            else:
                int1, err = spi.quad(integrand1, limits[0], limits[1])
                int2, err = spi.quad(integrand2, limits[0], limits[1])
            sigma_t_vals[i] = int1 / int2
    elif cs_method is CS_Method.point:
        for i in range(num_points):
            sigma_t_vals[i] = sigma_t.val(points[i])

    # Initialize arrays
    a = np.zeros((num_points, num_points), dtype=float)
    b = np.zeros((num_points), dtype=float)
    
    # Set matrix
    for i in range(num_points):
        for j in range(num_points):
            limits = weight.limits(j)
            if cs_method is CS_Method.full:
                def integrand(x):
                    bas = basis.val(i, x)
                    dbas = basis.dval(i, x)
                    wei = weight.val(j, x)
                    dwei = weight.dval(j, x)
                    ta = tau.val(x)
                    st = sigma_t.val(x)
                    return mu * (-bas + mu * ta * dbas) * dwei + st * bas * (wei + ta * mu * dwei)
            else:
                def integrand(x):
                    bas = basis.val(i, x)
                    dbas = basis.dval(i, x)
                    wei = weight.val(j, x)
                    dwei = weight.dval(j, x)
                    ta = tau.val(x)
                    st = sigma_t_vals[j]
                    return mu * (-bas + mu * ta * dbas) * dwei + st * bas * (wei + ta * mu * dwei)
            t1 = mu * basis.val(i, points[-1]) * weight.val(j, points[-1])
            if fixed_quadrature:
                t2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=int_ord)
            else:
                t2, abserr = spi.quad(integrand, limits[0], limits[1])
            a[j, i] = t1 + t2
            
    # Set RHS
    for j in range(num_points):
        limits = weight.limits(j)
        def integrand(x):
            return (weight.val(j, x) + tau.val(j) * weight.dval(j, x)) * source.val(x)

        t1 = mu * psi0 * weight.val(j, points[0])
        if fixed_quadrature:
            t2, err = spi.fixed_quad(vfunc, limits[0], limits[1], n=int_ord, args=[integrand])
        else:
            t2, abserr = spi.quad(integrand, limits[0], limits[1])
        b[j] = t1 + t2

    alpha = spl.solve(a, b)

    psi = np.zeros(num_points)
    for i in range(num_points):
        val = 0.
        for j in range(num_points):
            val += alpha[j] * basis.val(j, points[i])
        psi[i] = val

    analytic = np.zeros(num_points)
    for i in range(num_points):
        analytic[i] = solution.val(points[i])
    err = psi - analytic
    l2err = np.divide(np.sqrt(np.sum(np.power(err, 2))), 1. * len(err))

    return points, analytic, psi, err, l2err

if __name__ == '__main__':
    if len(sys.argv) != 15:
        print("supg_rbf_transport [basis weight cs_method num_points ep_basis ep_weight tau1 tau2 sigma1 sigma2 source1 source2 psi0]")
        sys.exit()
    i = itertools.count(1)
    basis = str(sys.argv[next(i)])
    weight = str(sys.argv[next(i)])
    cs_method = str(sys.argv[next(i)])
    fixed_quadrature = bool(int(sys.argv[next(i)]))
    num_points = int(sys.argv[next(i)])
    ep_basis = float(sys.argv[next(i)])
    ep_weight = float(sys.argv[next(i)])
    tau1 = float(sys.argv[next(i)])
    tau2 = float(sys.argv[next(i)])
    sigma1 = float(sys.argv[next(i)])
    sigma2 = float(sys.argv[next(i)])
    source1 = float(sys.argv[next(i)])
    source2 = float(sys.argv[next(i)])
    psi0 = float(sys.argv[next(i)])
    
    
    points, analytic, psi, err, l2err = supg_transport(basis,
                                                       weight,
                                                       CS_Method[cs_method],
                                                       num_points,
                                                       ep_basis,
                                                       ep_weight,
                                                       tau1,
                                                       tau2,
                                                       sigma1,
                                                       sigma2,
                                                       source1,
                                                       source2,
                                                       psi0,
                                                       fixed_quadrature)
    
    if True:
        description = ""
        for arg in sys.argv:
            description += arg + " "
        description += "l2err={:5e}".format(l2err)
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
    
