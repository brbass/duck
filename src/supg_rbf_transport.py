from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap
from enum import Enum

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

def get_limits(i,
               j,
               basis,
               weight):
    baslim = basis.limits(i)
    weilim = weight.limits(j)
    newlim = np.zeros(2)
    
    if baslim[1] < weilim[0] or baslim[0] > weilim[1]:
        return False, newlim
    newlim[0] = np.maximum(baslim[0], weilim[0])
    newlim[1] = np.minimum(baslim[1], weilim[1])

    return True, newlim
    
    

def supg_transport(basis_str,
                   weight_str,
                   cs_method,
                   quadrature_order,
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
                   mu,
                   plot_results = False):
    if quadrature_order == 0:
        fixed_quadrature = False
    else:
        fixed_quadrature = True
    # Get problem description
    description = "supg_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(basis_str,
                                                                          weight_str,
                                                                          cs_method.name,
                                                                          quadrature_order,
                                                                          num_points,
                                                                          ep_basis,
                                                                          ep_weight,
                                                                          tau1,
                                                                          tau2,
                                                                          sigma1,
                                                                          sigma2,
                                                                          source1,
                                                                          source2,
                                                                          psi0)
    
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    dx = points[1] - points[0]
    mu = 0.9
    
    # Set cross section and source
    sigma_t = Cross_Section(sigma1,
                            sigma2)
    source = Cross_Section(source1,
                           source2)
    
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
    elif basis_str == "compact_gaussian":
        basis = Compact_Gaussian(ep_basis,
                                 points)
    elif basis_str == "mls":
        polyord = 2
        num_neighbors = int(ep_basis)
        basis = MLS(polyord,
                    num_neighbors,
                    points)
    elif basis_str == "linear_mls":
        num_neighbors = int(ep_basis)
        basis = Linear_MLS(num_neighbors,
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
        polyord = 1
        num_neighbors = int(ep_weight)
        weight = MLS(polyord,
                     num_neighbors,
                     points)
    elif weight_str == "linear_mls":
        num_neighbors = int(ep_weight)
        weight = Linear_MLS(num_neighbors,
                            points)
    else:
        print("weight not found: " + weight_str)
        return

    # Set problem solution
    solution = Solution(sigma_t,
                        source,
                        psi0,
                        mu)

    # Set the SUPG parameter
    tau = Cross_Section(tau1 / weight.shape,
                        tau2 / weight.shape)
    tau_vals = np.zeros(num_points)
    for i in range(num_points):
        tau_vals[i] = tau.val(points[i])
    
    # Calculate sigma_t (if applicable)
    sigma_t_vals = np.zeros(num_points)
    if cs_method is CS_Method.full:
        pass
    elif cs_method is CS_Method.flux:
        for i in range(num_points):
            limits = weight.limits(i)
            ta = tau_vals[i]
            def integrand1(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                st = sigma_t.val(x)
                sol = solution.val(x)
                return (wei + ta * mu * dwei) * st * sol
            def integrand2(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                sol = solution.val(x)
                return (wei + ta * mu * dwei) * sol
            if fixed_quadrature:
                int1, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand1], n=quadrature_order)
                int2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand2], n=quadrature_order)
            else:
                int1, err = spi.quad(integrand1, limits[0], limits[1])
                int2, err = spi.quad(integrand2, limits[0], limits[1])
            sigma_t_vals[i] = int1 / int2
    elif cs_method is CS_Method.weight:
        for i in range(num_points):
            limits = weight.limits(i)
            ta = tau_vals[i]
            def integrand1(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                st = sigma_t.val(x)
                return (wei + ta * mu * dwei) * st
            def integrand2(x):
                wei = weight.val(i, x)
                dwei = weight.dval(i, x)
                return wei + ta * mu * dwei
            if fixed_quadrature:
                int1, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand1], n=quadrature_order)
                int2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand2], n=quadrature_order)
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
            nonzero, limits = get_limits(i,
                                         j,
                                         basis,
                                         weight)
            if nonzero:
                ta = tau_vals[j]
                if cs_method is CS_Method.full:
                    def integrand(x):
                        bas = basis.val(i, x)
                        dbas = basis.dval(i, x)
                        wei = weight.val(j, x)
                        dwei = weight.dval(j, x)
                        st = sigma_t.val(x)
                        return mu * (-bas + mu * ta * dbas) * dwei + st * bas * (wei + ta * mu * dwei)
                else:
                    def integrand(x):
                        bas = basis.val(i, x)
                        dbas = basis.dval(i, x)
                        wei = weight.val(j, x)
                        dwei = weight.dval(j, x)
                        st = sigma_t_vals[j]
                        return mu * (-bas + mu * ta * dbas) * dwei + st * bas * (wei + ta * mu * dwei)
                t1 = mu * basis.val(i, points[-1]) * weight.val(j, points[-1])
                if fixed_quadrature:
                    t2, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
                else:
                    t2, abserr = spi.quad(integrand, limits[0], limits[1])
                a[j, i] = t1 + t2
            else:
                a[j, i] = 0
    
    # Set RHS
    for j in range(num_points):
        limits = weight.limits(j)
        def integrand(x):
            return (weight.val(j, x) + tau_vals[j] * mu * weight.dval(j, x)) * source.val(x)

        t1 = mu * psi0 * weight.val(j, points[0])
        if fixed_quadrature:
            t2, err = spi.fixed_quad(vfunc, limits[0], limits[1], n=quadrature_order, args=[integrand])
        else:
            t2, abserr = spi.quad(integrand, limits[0], limits[1])
        b[j] = t1 + t2

    # Solve equation for coefficients
    alpha = spl.solve(a, b)
    
    # Calculate psi based on coefficients
    psi = np.zeros(num_points)
    for i in range(num_points):
        val = 0.
        for j in range(num_points):
            val += alpha[j] * basis.val(j, points[i])
        psi[i] = val

    # Get fine values of psi
    num_plot = 200
    x_plot = np.linspace(points[0], points[-1], num_plot, endpoint=True)
    psi_plot = np.zeros(num_plot)
    for i in range(num_plot):
        val = 0;
        for j in range(num_points):
            val += alpha[j] * basis.val(j, x_plot[i])
        psi_plot[i] = val

    # Get fine values of analytic solution
    analytic_plot = np.zeros(num_plot)
    for i in range(num_plot):
        analytic_plot[i] = solution.val(x_plot[i])
    err_plot = psi_plot - analytic_plot
    l2err = np.divide(np.sqrt(np.sum(np.power(err_plot, 2))), 1. * len(err_plot))
        
    # Get analytic solution
    analytic = np.zeros(num_points)
    for i in range(num_points):
        analytic[i] = solution.val(points[i])
    err = psi - analytic
    # l2err = np.divide(np.sqrt(np.sum(np.power(err, 2))), 1. * len(err))
    
    # Plot results
    if plot_results:
        description += "_l2err={:5e}".format(l2err)
        col = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ln1 = ax1.plot(x_plot, analytic_plot, label="analytic", color=col[0])
        ln2 = ax1.plot(x_plot, psi_plot, label="numeric", color=col[1])
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$\psi(x)$")
        ax1.grid()
        ln3 = ax2.plot(x_plot, err_plot, label="error", color=col[2])
        ax2.set_ylabel(r"$err(\psi(x))$")
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        plt.title("\n".join(wrap(description, 60)))
        plt.savefig("../figs/{}.pdf".format(description))
        plt.close()
        
    return points, analytic, psi, err, l2err

if __name__ == '__main__':
    if len(sys.argv) != 16:
        print("supg_rbf_transport [basis weight cs_method fixed_quadrature num_points ep_basis ep_weight tau1 tau2 sigma1 sigma2 source1 source2 psi0 mu]")
        sys.exit()
    i = itertools.count(1)
    basis = str(sys.argv[next(i)])
    weight = str(sys.argv[next(i)])
    cs_method = str(sys.argv[next(i)])
    quadrature_order = int(sys.argv[next(i)])
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
    mu = float(sys.argv[next(i)])
    
    points, analytic, psi, err, l2err = supg_transport(basis,
                                                       weight,
                                                       CS_Method[cs_method],
                                                       quadrature_order,
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
                                                       mu,
                                                       True)
