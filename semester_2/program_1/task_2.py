import numpy as np
import matplotlib.pyplot as plt
import interpolate_functions as ifs
import functions_1_to_4

"""
This task will perform the subtasks for the f_1 function:
    1.  The interpolating problem that the given polynomial solves on the uniform mesh points and Chebyshev
        points of the first and second kind, i.e., y_i = f(x_i), 0 <= i <= m where m = 9 for f_1(x).
    2.  For each of the degrees used for the uniform and Chebyshev meshes, determine the conditioning by
        evaluating kappa(x,n,y) and kappa(x,n,1) for a <= x <= b and summarize them appropriately using \Lambda_n
        and H_n along with appropriate statistics.
    3.  Assess the accuracy and stability of the single precision codes using the appropriate bounds from the
        notes and literature on the class webpage and the values generated in determining the conditioning in
        the subtask above. (The condition numbers and "exact" values of the interpolating polynomial for accuracy
        and stability assessment should be done in double precision.) This should be done for
            - Barycentric form 2 of the polynomial
            - Newton form with the mesh points in increasing order, decreasing order, and satisfying the Leja
            ordering conditions
        You should provide plots similar to those in Higham's paper for a small number of illustrative examples
        with 30 points for the uniform mesh and the Chebyshev points of the first kind. The other results should be
        summarized to comment on the accuracy and stability. Note that Higham's experiments are run in double precision
        and compared to "exact" values from Matlab's 50 digit symbolic arithmetic toolbox, so you will not see exactly
        the same behavior.
"""



def get_user_inputs():
    m = int(input("Please enter the number of mesh points (m) [default = 9]: ") or "9")
    rho = int(input("Please enter the root value, i.e. (x - 2) would enter 2 [default = 2]: ") or "2")
    d = int(input("Please enter the multiplicity of the root or highest degree of monomial [default = 9]: ") or "9")

    x_min = int(input("Please enter the minimum value for x values to be tested [default = -1]: ") or "-1")
    x_max = int(input("Please enter the maximum value for x values to be tested [default = 1]: ") or "1")

    num_tests = int(input("Please enter the number of tests wanted to run [default = 10]: ") or "10")

    return m, rho, d, x_min, x_max, num_tests


def task_2_driver():
    inputs = get_user_inputs()

    m, rho, d, x_min, x_max, num_tests = inputs

    # Setting up the initial mesh's using the user inputs
    x_mesh_uniform = np.linspace(-1, 1, m)
    x_mesh_cheb1 = ifs.chebyshev_points(x_min, x_max, m, flag=2, dtype=np.float32)
    x_mesh_cheb1_inc = ifs.x_mesh_order(x_mesh_cheb1, 2)
    x_mesh_cheb2 = ifs.chebyshev_points(x_min, x_max, m, flag=3, dtype=np.float32)
    #f = functions_1_to_4.p_1(d, rho)
    f = functions_1_to_4.p_4(2)

    bary_2_sol = []


    for i in range(num_tests):
        x_values = np.linspace(x_min, x_max, 100)
        exact = f(x_values)

        ### CHEBYSHEV 1 POINTS

        # Barycentric Form 1 Approximate (float32)
        g, func_val_1 = ifs.coef_gamma(x_mesh_cheb1_inc, m, f, dtype=np.float32)
        bary_1 = ifs.bary_1_interpolation(g, x_mesh_cheb1_inc, x_values, func_val_1, m, dtype=np.float32)

        #print(bary_1)

        # Barycentric Form 1 Exact (float64)


        # Barycentric Form 1 Error


        # Barycentric Form 2 Approximate (float32)
        c, func_val_2 = ifs.coef_beta(x_mesh_cheb1_inc, m, f, flag=2, dtype=np.float32)
        bary_2 = ifs.bary_2_interpolation(c, x_mesh_cheb1_inc, x_values, func_val_2, m, dtype=np.float32)

        error = np.subtract(exact, bary_1)
        error = abs(error)
        rel_error = np.divide(error, exact)

        # Barycentric Form 2 Exact (float64)




        # Plot Barycentric Form
        plt.title("Barycentric Form 1 and 2 with Chebyshev Points of the First Kind")
        plt.plot(x_mesh_cheb1_inc, func_val_2, '*', label="Interpolation Points")
        #plt.plot(x_values, bary_2, label="Barycentric 2")
        plt.plot(x_values, bary_1, label="Barycentric 1")
        plt.plot(x_values, exact, 'black', label="f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.title("Error")
        plt.plot(x_values, rel_error, 'x', label="Error")
        plt.grid(True)
        plt.show()

        bary_2_sol.append(bary_2)





    return bary_2_sol

if __name__ == '__main__':
    while True:
        bary_2_sol = task_2_driver()

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("\nThank you for using the Interpolation Driver!")
