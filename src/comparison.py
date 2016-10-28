import numpy as np
from matplotlib import pyplot as plt

from supg_rbf_transport import supg_transport, CS_Method
from weak_rbf_transport import weak_transport
from strong_rbf_transport import strong_transport

def compare():
    basis = "multiquadric"
    weight = "compact_gaussian"
    num_points_vals = np.array([2**i for i in range(3, 9)])
    num_vals = len(num_points_vals)
    ep_basis = 0.5
    ep_weight = 1.0
    tau1 = 0.3
    tau2 = 0.3
    sigma1 = 1.0
    sigma2 = 10.0
    source1 = 1.0
    source2 = 0.0
    psi0 = 0.0
    fixed_quadrature = False
    plot_results = True
    
    run_problem = True
    cases = ['strong', 'weak', 'supg_full', 'supg_flux', 'supg_weight', 'supg_point']
    num_cases = len(cases)
    err = np.zeros((num_vals, num_cases))
    
    description = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(basis, weight, ep_basis, ep_weight, tau1, tau2, sigma1, sigma2, source1, source2, psi0)
    filename = "../out/{}.txt".format(description)
    
    if run_problem:
        for i, num_points in enumerate(num_points_vals):
            for j, case in enumerate(cases):
                if case == 'strong':
                    temp1, temp2, temp3, temp4, l2err = strong_transport(basis,
                                                                         num_points,
                                                                         ep_basis,
                                                                         sigma1,
                                                                         sigma2,
                                                                         source1,
                                                                         source2,
                                                                         psi0,
                                                                         plot_results)
                    err[i, j] = l2err
                elif case == 'weak':
                    temp1, temp2, temp3, temp4, l2err = supg_transport(basis,
                                                                       weight,
                                                                       CS_Method.full,
                                                                       num_points,
                                                                       ep_basis,
                                                                       ep_weight,
                                                                       0.0,
                                                                       0.0,
                                                                       sigma1,
                                                                       sigma2,
                                                                       source1,
                                                                       source2,
                                                                       psi0,
                                                                       fixed_quadrature,
                                                                       plot_results)
                    err[i, j] = l2err
                elif case == 'supg_full':
                    temp1, temp2, temp3, temp4, l2err = supg_transport(basis,
                                                                       weight,
                                                                       CS_Method.full,
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
                                                                       fixed_quadrature,
                                                                       plot_results)
                    err[i, j] = l2err
                elif case == 'supg_flux':
                    temp1, temp2, temp3, temp4, l2err = supg_transport(basis,
                                                                       weight,
                                                                       CS_Method.flux,
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
                                                                       fixed_quadrature,
                                                                       plot_results)
                    err[i, j] = l2err
                elif case == 'supg_weight':
                    temp1, temp2, temp3, temp4, l2err = supg_transport(basis,
                                                                       weight,
                                                                       CS_Method.weight,
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
                                                                       fixed_quadrature,
                                                                       plot_results)
                    err[i, j] = l2err
                elif case == 'supg_point':
                    temp1, temp2, temp3, temp4, l2err = supg_transport(basis,
                                                                       weight,
                                                                       CS_Method.point,
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
                                                                       fixed_quadrature,
                                                                       plot_results)
                    err[i, j] = l2err
                else:
                    print("case not found: " + case)
                    
        np.savetxt(filename, err)
    else:
        err = np.loadtxt(filename)
    
    plt.figure()
    for i, label in enumerate(cases):
        plt.loglog(num_points_vals, err[:, i], marker="o", label=label, basex=2, basey=10)
    plt.xlabel("number of points")
    plt.ylabel(r"$L_2$ error")
    plt.legend()
    plt.savefig("../figs/{}.pdf".format(description))
    plt.close()

if __name__ == '__main__':
    compare()
