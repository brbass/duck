from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap
import argparse

# Vector support for scalar function
def vfunc(xvals, func):
    for i, x in enumerate(xvals):
        xvals[i] = func(x)
    return xvals

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

def solution(k1,
             k2,
             h,
             q1,
             q2,
             tinf,
             r):
    rr = 2
    rm = 1
    if r <= rm:
        return (2*pow(k2,2)*q1*pow(rm,2) - 2*k1*k2*q2*pow(rm,2) - h*k2*q1*pow(r,2)*rr + h*k2*q1*pow(rm,2)*rr - h*k1*q2*pow(rm,2)*rr + 2*k1*k2*q2*pow(rr,2) + h*k1*q2*pow(rr,3) + 4*h*k1*k2*rr*tinf + 2*h*(-(k2*q1) + k1*q2)*pow(rm,2)*rr*np.log(rm) + 2*h*(k2*q1 - k1*q2)*pow(rm,2)*rr*np.log(rr))/(4.*h*k1*k2*rr)
    else: 
        return (2*pow(k2,2)*q1*pow(rm,2) - 2*k1*k2*q2*pow(rm,2) - h*k1*q2*pow(r,2)*rr + 2*k1*k2*q2*pow(rr,2) + h*k1*q2*pow(rr,3) + 4*h*k1*k2*rr*tinf + 2*h*(-(k2*q1) + k1*q2)*pow(rm,2)*rr*np.log(r) + 2*h*(k2*q1 - k1*q2)*pow(rm,2)*rr*np.log(rr))/(4.*h*k1*k2*rr)

def heat_transfer(bastype,
                  basep,
                  weitype,
                  weiep,
                  quadrature_order,
                  num_points,
                  cond1,
                  cond2,
                  conv,
                  src1,
                  src2,
                  tinf,
                  plot_results = False):
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    dr = points[1] - points[0]
    
    # Initialize data
    if quadrature_order == 0:
        fixed_quadrature = False
    else:
        fixed_quadrature = True
    basis = get_basis(bastype,
                      points,
                      basep)
    weight = get_basis(weitype,
                       points,
                       weiep)
    conductivity = Cross_Section(cond1, cond2)
    source = Cross_Section(src1, src2)
    
    # Get problem description
    description = "cyl_heat_{}_{}_{}_{}_{}_{}".format(basis.description(),
                                                      weight.description(),
                                                      quadrature_order,
                                                      num_points,
                                                      conductivity.description(),
                                                      source.description())
    
    # Get matrix
    matrix = np.zeros((num_points, num_points), dtype=float)
    for j in range(num_points):
        for i in range(num_points):
            nonzero, limits = get_limits(i, j, basis, weight)

            if nonzero:
                def integrand(r):
                    dwei = weight.dval(j, r)
                    dbas = basis.dval(i, r)
                    cond = conductivity.val(r)
                    return dwei * dbas * cond * r
                if fixed_quadrature:
                    val, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
                else:
                    val, err = spi.quad(integrand, limits[0], limits[1])
                matrix[j, i] = val
            else:
                matrix[j, i] = 0
                
    # Get source
    rhs = np.zeros((num_points), dtype=float)
    for j in range(num_points):
        def integrand(r):
            return source.val(r) * weight.val(j, r) * r
        limits = weight.limits(j)
        if fixed_quadrature:
            val, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
        else:
            val, err = spi.quad(integrand, limits[0], limits[1])
        rhs[j] = val
        
    # Get boundary term
    for j in range(num_points):
        wei = weight.val(j, points[-1])
        for i in range(num_points):
            bas = basis.val(i, points[-1])
            matrix[j, i] += conv * wei * bas * points[-1]
        rhs[j] += wei * conv * tinf * points[-1]
    
    # Solve equation for coefficients
    coefficients = spl.solve(matrix, rhs)
    
    # Calculate temperature at plot points
    num_plot = 200
    x_plot = np.linspace(points[0], points[-1], num_plot, endpoint=True)
    t_plot = np.zeros(num_plot)
    t_ana = np.zeros(num_plot)
    for i in range(num_plot):
        val = 0
        for j in range(num_points):
            val += coefficients[j] * basis.val(j, x_plot[i])
        t_plot[i] = val
        t_ana[i] = solution(cond1,
                            cond2,
                            conv,
                            src1,
                            src2,
                            tinf,
                            x_plot[i])
    err_plot = t_plot - t_ana
    l2err = np.divide(np.sqrt(np.sum(np.power(err_plot, 2))), 1. * len(err_plot))
    
    # Plot results
    if plot_results:
        description += "_l2err={:5e}".format(l2err)
        col = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ln1 = ax1.plot(x_plot, t_ana, label="analytic", color=col[0])
        ln2 = ax1.plot(x_plot, t_plot, label="numeric", color=col[1])
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$T(r)$")
        ax1.grid()
        ln3 = ax2.plot(x_plot, err_plot, label="error", color=col[2])
        ax2.set_ylabel(r"$err(T(r))$")
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        plt.title("\n".join(wrap(description, 60)))
        # plt.savefig("../figs/{}.pdf".format(description))
        # plt.close()
        plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bas", type=str, nargs=2, required=True,
                        help="basis/weight function types")
    parser.add_argument("--ep", type=float, nargs=2, required=True,
                        help="basis/weight function shape")
    parser.add_argument("--points", type=int, required=True,
                        help="number of points")
    parser.add_argument("--cond", type=int, nargs=2, required=True,
                        help="conduction coefficient")
    parser.add_argument("--conv", type=int, required=True,
                        help="convection coefficient")
    parser.add_argument("--src", type=float, nargs=2, required=True,
                        help="internal source strength")
    parser.add_argument("--tinf", type=float, required=True,
                        help="convection fluid temperature")
    parser.add_argument("--quadord", type=int, default=0,
                        help="quadrature order (0 = adaptive)")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="show plot")
    args = parser.parse_args()
    
    heat_transfer(args.bas[0],
                  args.ep[0],
                  args.bas[1],
                  args.ep[1],
                  args.quadord,
                  args.points,
                  args.cond[0],
                  args.cond[1],
                  args.conv,
                  args.src[0],
                  args.src[1],
                  args.tinf,
                  args.plot)
