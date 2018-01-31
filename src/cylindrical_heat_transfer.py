from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap
import argparse
import scipy.sparse as sps

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
    rm = 1.
    rr = 2.
    if r <= rm:
        return (2*k1*k2*q1*pow(rm,2) - 2*k1*k2*q2*pow(rm,2) - h*k2*q1*pow(r,2)*rr + h*k2*q1*pow(rm,2)*rr - h*k1*q2*pow(rm,2)*rr + 2*k1*k2*q2*pow(rr,2) + h*k1*q2*pow(rr,3) + 4*h*k1*k2*rr*tinf + 2*h*k1*(-q1 + q2)*pow(rm,2)*rr*np.log(rm) + 2*h*k1*(q1 - q2)*pow(rm,2)*rr*np.log(rr))/(4.*h*k1*k2*rr)
    else:
        return (2*k2*q1*pow(rm,2) - 2*k2*q2*pow(rm,2) - h*q2*pow(r,2)*rr + 2*k2*q2*pow(rr,2) + h*q2*pow(rr,3) + 4*h*k2*rr*tinf + 2*h*(-q1 + q2)*pow(rm,2)*rr*np.log(r) + 2*h*(q1 - q2)*pow(rm,2)*rr*np.log(rr))/(4.*h*k2*rr)

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
    description = "rbf_cyl_heat_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(bastype,
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
                                                                            tinf)
    
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
    if basis.local() and weight.local():
        matrix = sps.csr_matrix(matrix)
        coefficients = sps.linalg.spsolve(matrix, rhs)
    else:
        coefficients = spl.solve(matrix, rhs)
    
    # Calculate temperature at plot points
    num_plot = 1001
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
        make_plot(l2err,
                  x_plot,
                  t_ana,
                  t_plot,
                  err_plot,
                  description)

def cfem_heat_transfer(order,
                       num_points,
                       cond1,
                       cond2,
                       conv,
                       src1,
                       src2,
                       tinf,
                       plot_results = False):
    # Get description
    description = "cfem_cyl_heat_{}_{}_{}_{}_{}_{}_{}_{}".format(order,
                                                                 num_points,
                                                                 cond1,
                                                                 cond2,
                                                                 conv,
                                                                 src1,
                                                                 src2,
                                                                 tinf)
    
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)

    # Get basis and weight functions
    basis = Lagrange(order,
                     points)
    weight = basis
    num_cells = basis.num_cells
    num_nodes = basis.num_nodes
    num_unknowns = basis.num_cells * (basis.num_nodes - 1) + 1
    quadrature_order = 8
    
    # Initialize data
    conductivity = Cross_Section(cond1, cond2)
    source = Cross_Section(src1, src2)
    
    # Get matrix
    matrix = np.zeros((num_unknowns, num_unknowns), dtype=float)
    for i in range(num_cells):
        limits = [points[i+1], points[i]]
        
        for j in range(num_nodes):
            for k in range(num_nodes):
                def integrand(r):
                    dwei = weight.dval(i, j, r)
                    dbas = basis.dval(i, k, r)
                    cond = conductivity.val(r)
                    return dwei * dbas * cond * r
                val, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
                m = j + i * (num_nodes - 1)
                n = k + i * (num_nodes - 1)
                matrix[m, n] += val
                
    # Get source
    rhs = np.zeros((num_unknowns), dtype=float)
    for i in range(num_cells):
        limits = [points[i+1], points[i]]
        for j in range(num_nodes):
            def integrand(r):
                return source.val(r) * weight.val(i, j, r) * r
            val, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
            m = j + i * (num_nodes - 1)
            rhs[m] += val

    # Get boundary term
    i = num_cells - 1
    for j in range(num_nodes):
        m = j + i * (num_nodes - 1)
        wei = weight.val(i, j, points[-1])
        rhs[m] += wei * conv * tinf * points[-1]
        for k in range(num_nodes):
            bas = basis.val(i, k, points[-1])
            n = k + i * (num_nodes - 1)
            matrix[m, n] += conv * wei * bas * points[-1]

    # Solve equation for coefficients
    matrix = sps.csr_matrix(matrix)
    lhs = sps.linalg.spsolve(matrix, rhs)
    coefficients = np.zeros((num_cells, num_nodes), dtype=float)
    for i in range(num_cells):
        for j in range(num_nodes):
            k = j + i * (num_nodes - 1)
            coefficients[i, j] = lhs[k]
    
    # Calculate temperature at plot points
    num_plot = 201
    x_plot = np.linspace(points[0], points[-1], num_plot, endpoint=True)
    t_plot = np.zeros((num_plot))
    t_ana = np.zeros((num_plot))
    for i in range(num_plot):
        x = x_plot[i]
        val = 0
        k = basis.get_cell(x)
        for j in range(num_nodes):
            val += coefficients[k, j] * basis.val(k, j, x)
        t_plot[i] = val
        t_ana[i] = solution(cond1,
                            cond2,
                            conv,
                            src1,
                            src2,
                            tinf,
                            x)
    err_plot = t_plot - t_ana
    l2err = np.divide(np.sqrt(np.sum(np.power(err_plot, 2))), 1. * len(err_plot))
    
    # Plot results
    if plot_results:
        make_plot(l2err,
                  x_plot,
                  t_ana,
                  t_plot,
                  err_plot,
                  description)
        
def make_plot(l2err,
              x_plot,
              t_ana,
              t_plot,
              err_plot,
              description):
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
    print("displaying plot")
    plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='option')

    # Get meshless options
    parser_rbf = subparsers.add_parser("rbf",
                                       help="run in rbf mode")
    parser_rbf.add_argument("--bas", type=str, nargs=2, required=True,
                        help="basis/weight function types")
    parser_rbf.add_argument("--ep", type=float, nargs=2, required=True,
                            help="basis/weight function shape")
    parser_rbf.add_argument("--quadord", type=int, default=0,
                            help="quadrature order (0 = adaptive)")
    
    # Get CFEM options
    parser_cfem = subparsers.add_parser("cfem",
                                        help="run in cfem mode")
    parser_cfem.add_argument("--ord", type=int,
                             help="lagrange polynomial order")
    
    # Add shared options
    for p in [parser_rbf, parser_cfem]:
        p.add_argument("--points", type=int, required=True,
                       help="number of points")
        p.add_argument("--cond", type=float, nargs=2, required=True,
                       help="conduction coefficient")
        p.add_argument("--conv", type=float, required=True,
                       help="convection coefficient")
        p.add_argument("--src", type=float, nargs=2, required=True,
                       help="internal source strength")
        p.add_argument("--tinf", type=float, required=True,
                       help="convection fluid temperature")
        p.add_argument("--plot", action='store_true', default=False,
                       help="show plot")
        
    # Get arguments and run
    args = parser.parse_args()

    if args.option == "rbf":
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
    elif args.option == "cfem":
        cfem_heat_transfer(args.ord,
                           args.points,
                           args.cond[0],
                           args.cond[1],
                           args.conv,
                           args.src[0],
                           args.src[1],
                           args.tinf,
                           args.plot)
    else:
        print("command {} not found".format(args.option))
