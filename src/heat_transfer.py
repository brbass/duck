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

def heat_transfer(basis,
                  weight,
                  quadrature_order,
                  num_points,
                  conductivity,
                  source,
                  plot_results = False):
    if quadrature_order == 0:
        fixed_quadrature = False
    else:
        fixed_quadrature = True

    # Get problem description
    description = "heat_{}_{}_{}_{}_{}_{}".format(basis.description(),
                                                  weight.description(),
                                                  quadrature_order,
                                                  num_points,
                                                  conductivity.description(),
                                                  source.description())
    
    # Initialize geometry
    length = 2
    points = np.linspace(0, length, num_points)
    dx = points[1] - points[0]
    
    # Get matrix
    matrix = np.zeros((num_points, num_points), dtype=float)
    for j in range(num_points):
        for i in range(num_points):
            nonzero, limits = get_limits(i, j, basis, weight)

            if nonzero:
                def integrand(x):
                    wei = weight.dval(j, x)
                    bas = basis.dval(i, x)
                    con = conductivity.val(x)
                    return wei * bas * con
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

    # Solve equation for coefficients
    coefficients = spl.solve(matrix, rhs)

    # Calculate temperature based on coefficients
    temp = np.zeros(num_points)
    for i in range(num_points):
        val = 0
        for j in range(num_points):
            val += coefficients[j] * basis.val(j, points[i])
        temp[i] = val

    if plot_results:
        plt.plot(points, temp)
        plt.show()

def run_test():
    length = 2
    num_points = 40
    num_neighbors = 4
    points = np.linspace(0, length, num_points)
    basis = get_basis("linear-mls",
                      points,
                      num_neighbors)
    weight = get_basis("linear-mls",
                       points,
                       num_neighbors)
    quad_order = 0
    conductivity = Cross_Section(1, 1)
    source = Cross_Section(1, 1)
    plot = True
    heat_transfer(basis,
                  weight,
                  quad_order,
                  num_points,
                  conductivity,
                  source,
                  plot)
    
if __name__ == '__main__':
    run_test()
