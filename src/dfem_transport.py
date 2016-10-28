from two_region import *
import numpy as np
import scipy.linalg as spl
import scipy.integrate as spi
import sys
import itertools
from textwrap import wrap

def dfem_transport(half_elements,
                   sigma1,
                   sigma2,
                   source1,
                   source2,
                   psi0,
                   plot_results = False):
    # Get problem description
    description = "dfem_{}_{}_{}_{}_{}_{}".format(half_elements,
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
    mu = 1
    
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
    
    for i in range(num_elements):
        
    
    
    # Remove once code is done
    points = 0
    analytic = 0
    psi = 0
    err = 0
    l2err = 0
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
    
            
    
