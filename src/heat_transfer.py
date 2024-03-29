from basis import *
from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap

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
             h1,
             h2,
             q1,
             q2,
             tinf1,
             tinf2,
             x):
    xl = 0
    xr = 2
    xm = 1
    if x <= xm:
        return (k1*(2*k2*(k2*q1*(-xl + xm) + k1*q2*(-xm + xr)) + h2*(k1*(2*k2*tinf2 + q2*pow(xm - xr,2)) - k2*q1*(pow(x,2) - 2*x*xl + pow(xm,2) + 2*xl*xr - 2*xm*xr))) + h1*(-(k2*q1*(x - xl)*(k2*(x + xl - 2*xm) + h2*(pow(xm,2) + xl*xr - 2*xm*xr + x*(-xl + xr)))) + k1*(2*pow(k2,2)*tinf1 + h2*q2*(x - xl)*pow(xm - xr,2) - 2*k2*(q2*(x - xl)*(xm - xr) + h2*(tinf1*x - tinf2*x + tinf2*xl - tinf1*xr)))))/(2.*k1*k2*(h1*k2 + h2*(k1 + h1*(-xl + xr))))
    else:
        return (k1*(2*k2*(k2*q1*(-xl + xm) + k1*q2*(-xm + xr)) + h2*(2*k2*q1*(xl - xm)*(x - xr) + k1*(2*k2*tinf2 - q2*(x - xr)*(x - 2*xm + xr)))) + h1*(k2*q1*pow(xl - xm,2)*(k2 + h2*(-x + xr)) + k1*(2*pow(k2,2)*tinf1 + h2*q2*(x - xr)*(-2*xl*xm + pow(xm,2) + x*(xl - xr) + xl*xr) - k2*(2*h2*(tinf1*x - tinf2*x + tinf2*xl - tinf1*xr) + q2*(pow(x,2) - 2*xl*xm + pow(xm,2) - 2*x*xr + 2*xl*xr)))))/(2.*k1*k2*(h1*k2 + h2*(k1 + h1*(-xl + xr))))

def heat_transfer(bastype,
                  basep,
                  weitype,
                  weiep,
                  quadrature_order,
                  num_points,
                  cond1,
                  cond2,
                  conv1,
                  conv2,
                  src1,
                  src2,
                  tinf1,
                  tinf2,
                  plot_results = False):
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    dx = points[1] - points[0]
    
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
    convection = Cross_Section(conv1, conv2)
    source = Cross_Section(src1, src2)
    tempinf = Cross_Section(tinf1, tinf2)
    
    # Get problem description
    description = "heat_{}_{}_{}_{}_{}_{}".format(basis.description(),
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
                def integrand(x):
                    wei = weight.dval(j, x)
                    bas = basis.dval(i, x)
                    cond = conductivity.val(x)
                    return wei * bas * cond
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
        def integrand(x):
            return source.val(x) * weight.val(j, x)
        limits = weight.limits(j)
        if fixed_quadrature:
            val, err = spi.fixed_quad(vfunc, limits[0], limits[1], args=[integrand], n=quadrature_order)
        else:
            val, err = spi.quad(integrand, limits[0], limits[1])
        rhs[j] = val

    # Get boundary terms
    conv1 = convection.val(points[0])
    conv2 = convection.val(points[-1])
    for j in range(num_points):
        wei1 = weight.val(j, points[0])
        wei2 = weight.val(j, points[-1])
        for i in range(num_points):
            bas1 = basis.val(i, points[0])
            bas2 = basis.val(i, points[-1])
            matrix[j, i] += conv1 * wei1 * bas1 + conv2 * wei2 * bas2
        tinf1 = tempinf.val(points[0])
        tinf2 = tempinf.val(points[-1])
        rhs[j] += wei1 * conv1 * tinf1 + wei2 * conv2 * tinf2
    
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
                            conv1,
                            conv2,
                            src1,
                            src2,
                            tinf1,
                            tinf2,
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
        ax1.set_ylabel(r"$T(x)$")
        ax1.grid()
        ln3 = ax2.plot(x_plot, err_plot, label="error", color=col[2])
        ax2.set_ylabel(r"$err(T(x))$")
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)
        plt.title("\n".join(wrap(description, 60)))
        # plt.savefig("../figs/{}.pdf".format(description))
        # plt.close()
        plt.show()
        
if __name__ == '__main__':
    if (len(sys.argv) != 15):
        print("heat_transfer [bastype basep weitype weiep quad_order num_points cond1 cond2 conv1 conv2 src1 src2 tinf1 tinf2]")
        sys.exit()
    i = itertools.count(1)
    bastype = str(sys.argv[next(i)])
    basep = float(sys.argv[next(i)])
    weitype = str(sys.argv[next(i)])
    weiep = float(sys.argv[next(i)])
    quad_order = int(sys.argv[next(i)])
    num_points = int(sys.argv[next(i)])
    cond1 = float(sys.argv[next(i)])
    cond2 = float(sys.argv[next(i)])
    conv1 = float(sys.argv[next(i)])
    conv2 = float(sys.argv[next(i)])
    src1 = float(sys.argv[next(i)])
    src2 = float(sys.argv[next(i)])
    tinf1 = float(sys.argv[next(i)])
    tinf2 = float(sys.argv[next(i)])
    heat_transfer(bastype,
                  basep,
                  weitype,
                  weiep,
                  quad_order,
                  num_points,
                  cond1,
                  cond2,
                  conv1,
                  conv2,
                  src1,
                  src2,
                  tinf1,
                  tinf2,
                  True)
