from two_region import *
import numpy as np
import scipy.linalg as spl
import sys
import itertools
from textwrap import wrap
from matplotlib import pyplot as plt

def dfem_transport(half_elements,
                   sigma1,
                   sigma2,
                   source1,
                   source2,
                   psi0,
                   mu,
                   plot_results = False):
    # Get problem description
    description = "dfem_{}_{}_{}_{}_{}_{}".format(half_elements*2,
                                                  sigma1,
                                                  sigma2,
                                                  source1,
                                                  source2,
                                                  psi0)
    
    # Initialize geometry
    num_nodes = 2
    num_elements = 2 * half_elements
    length = 2
    cont_points = np.linspace(0, length, num_elements + 1, endpoint=True)
    points = np.zeros(num_elements * num_nodes)
    for i in range(num_elements):
        for n in range(num_nodes):
            points[n + num_nodes * i] = cont_points[n + i]
    dx = points[1] - points[0]
    midpoints = np.linspace(0.5*dx, length-0.5*dx, num_elements, endpoint=True)
    
    # Set cross section and source
    sigma_t = Cross_Section(sigma1,
                            sigma2)
    source = Cross_Section(source1,
                           source2)

    sigma_t_vals = np.zeros(num_elements)
    source_vals = np.zeros(num_elements)
    for i in range(num_elements):
        sigma_t_vals[i] = sigma_t.val(midpoints[i])
        source_vals[i] = source.val(midpoints[i])
        
    # Sweep
    psi = np.zeros(num_elements * num_nodes)
    psi[0] = psi0

    mat1 = np.array([[1, 1],
                     [-1, 1]])
    mat2 = np.array([[1, 0],
                     [0, 1]])
    vec1 = np.array([1, 1])
    vec2 = np.array([1, 0])
    upstream = 0
    for i in range(num_elements):
        mat = mu * mat1 + dx * sigma_t_vals[i] * mat2
        lhs = dx * vec1 * source_vals[i] + 2 * mu * vec2 * psi[upstream]
        res = spl.solve(mat, lhs)
        psi[0 + num_nodes * i] = res[0]
        psi[1 + num_nodes * i] = res[1]
        upstream = 1 + num_nodes * i

    # Get analytic solution
    solution = Solution(sigma_t,
                        source,
                        psi0,
                        mu)
    analytic = np.zeros(num_elements * num_nodes)
    for i in range(num_elements * num_nodes):
        analytic[i] = solution.val(points[i])
    err = psi - analytic
    l2err= np.divide(np.sqrt(np.sum(np.power(err, 2))), 1. * len(err))
        
    # Plot results
    if plot_results:
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
    if len(sys.argv) != 7:
        print("dfem_transport [half_elements, sigma1, sigma2, source1, source2, psi0]")
        sys.exit()
    i = itertools.count(1)
    half_elements = int(sys.argv[next(i)])
    sigma1 = float(sys.argv[next(i)])
    sigma2 = float(sys.argv[next(i)])
    source1 = float(sys.argv[next(i)])
    source2 = float(sys.argv[next(i)])
    psi0 = float(sys.argv[next(i)])
    
    points, analytic, psi, err, l2err = dfem_transport(half_elements,
                                                       sigma1,
                                                       sigma2,
                                                       source1,
                                                       source2,
                                                       psi0,
                                                       True)
    
            
    
