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



# def get_user_inputs():
#     m = int(input("Please enter the number of mesh points (m) [default = 9]: ") or "9")
#     rho = int(input("Please enter the root value, i.e. (x - 2) would enter 2 [default = 2]: ") or "2")
#     d = int(input("Please enter the multiplicity of the root or highest degree of monomial [default = 9]: ") or "9")
#
#     a = float(input("Please enter the lower bound on the interval for mesh points (float) [default = -1]: ") or "-1")
#     b = float(input("Please enter the upper bound on the interval for mesh points (float) [default = 1]: ") or "1")
#
#     x_min = float(input("Please enter the minimum value for x values to be tested (float) [default = -1]: ") or "-1")
#     x_max = float(input("Please enter the maximum value for x values to be tested (float) [default = 1]: ") or "1")
#
#     num_tests = int(input("Please enter the number of tests wanted to run [default = 10]: ") or "10")
#
#     return m, rho, d, a, b, x_min, x_max, num_tests
#
#
# def task_2_driver():
#     inputs = get_user_inputs()
#
#     m, rho, d, a, b, x_min, x_max, num_tests = inputs
#
#     # Setting up the initial mesh's using the user inputs
#     x_mesh_uniform = np.linspace(-1, 1, m)
#     x_mesh_cheb1 = ifs.chebyshev_points(a, b, m, flag=2, dtype=np.float32)
#     x_mesh_cheb1_inc = ifs.x_mesh_order(x_mesh_cheb1, 2)
#     x_mesh_cheb2 = ifs.chebyshev_points(a, b, m, flag=3, dtype=np.float32)
#     f = functions_1_to_4.p_1(d, rho)
#     #f = functions_1_to_4.p_4(2)
#
#
#     x_mesh_cheb1_exact = ifs.chebyshev_points(a, b, m, flag=2, dtype=np.float64)
#     x_mesh_cheb1_exact_inc = ifs.x_mesh_order(x_mesh_cheb1_exact, 2)
#     x_mesh_cheb2_exact = ifs.chebyshev_points(a, b, m, flag=3, dtype=np.float64)
#     x_mesh_cheb2_exact_inc = ifs.x_mesh_order(x_mesh_cheb2_exact, 2)
#
#     bary_2_sol = []
#
#
#     for i in range(num_tests):
#         x_values = np.linspace(x_min, x_max, 100)
#         exact = f(x_values)
#
#         ### CHEBYSHEV 1 POINTS
#
#         # Barycentric Form 1 Approximate (float32)
#         g, func_val_1 = ifs.coef_gamma(x_mesh_cheb1_inc, f, dtype=np.float32)
#         bary_1, cond_1_b1_32, cond_y_numer_b1_32 = ifs.bary_1_interpolation(g, x_mesh_cheb1_inc, x_values, func_val_1, dtype=np.float32)
#
#         #print(bary_1)
#
#         # Barycentric Form 1 Exact (float64)
#         g_exact, func_val_1_exact = ifs.coef_gamma(x_mesh_cheb1_exact_inc, f, dtype=np.float64)
#         bary_1_exact, cond_1_b1_64, cond_y_numer_b1_64 = ifs.bary_1_interpolation(g_exact, x_mesh_cheb1_exact_inc, x_values, func_val_1_exact, dtype=np.float64)
#
#
#         # Barycentric Form 1 Error
#         error = np.subtract(bary_1_exact, bary_1)
#         error = abs(error)
#         rel_error = np.divide(error, bary_1_exact)
#
#         pointwise_bary1 = np.subtract(bary_1, exact)
#         pointwise_bary1 = abs(pointwise_bary1)
#
#         # Barycentric Form 2 Approximate (float32)
#         c, func_val_2 = ifs.coef_beta(x_mesh_cheb1, f, flag=2, dtype=np.float32)
#         bary_2, cond_1_b2_32, cond_y_b2_32 = ifs.bary_2_interpolation(c, x_mesh_cheb1, x_values, func_val_2, dtype=np.float32)
#
#
#         # Barycentric Form 2 Exact (float64)
#         c_exact, func_val_2_exact = ifs.coef_beta(x_mesh_cheb1_exact_inc, f, flag=2, dtype=np.float64)
#         bary_2_exact, cond_1_b2_64, cond_y_b2_64 = ifs.bary_2_interpolation(c_exact, x_mesh_cheb1_exact_inc, x_values, func_val_2_exact, dtype=np.float64)
#
#         # Barycentric Form 2 Error
#         err_bary_2 = np.subtract(bary_2_exact, bary_2)
#         err_bary_2 = abs(err_bary_2)
#         rel_err_bary_2 = np.divide(err_bary_2, bary_2_exact)
#
#         pointwise_bary2 = np.subtract(bary_2, exact)
#         pointwise_bary2 = abs(pointwise_bary2)
#
#         # Newton Divided Difference (float32)
#         func_val_newt_32, div_32 = ifs.newton_divdiff(x_mesh_cheb1_inc, f, dtype=np.float32)
#         newt_32 = ifs.horner_interpolation(x_mesh_cheb1_inc, x_values, div_32, dtype=np.float32)
#
#         # Plot Barycentric Form
#         plt.title("Barycentric Form 1 and 2 with Chebyshev Points of the First Kind")
#         plt.plot(x_mesh_cheb1, func_val_2, '*', label="Interpolation Points")
#         plt.plot(x_values, bary_2, label="Barycentric 2")
#         #plt.plot(x_values, newt_32, label="Newton's Method")
#         #plt.plot(x_values, bary_1, label="Barycentric 1")
#         plt.plot(x_values, exact, 'black', label="f(x)")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#
#         plt.title("Relative Error Between 'Exact' and 'Approximate' Solutions")
#         #plt.yscale('log')
#         plt.plot(x_values, rel_error, 'x', color='black', label="Barycentric Form 1")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.show()
#
#         plt.title("Relative Error Between 'Exact' and 'Approximate' Solutions")
#         plt.plot(x_values, rel_err_bary_2, 'x', color='black', label="Barycentric Form 2")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.show()
#
#
#         plt.title("Pointwise Error between f(x) and Barycentric Form 1")
#         #plt.yscale('log')
#         plt.plot(x_values, pointwise_bary1, color='black', label="Barycentric Form 1")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.show()
#
#         plt.title("Pointwise Error between f(x) and Barycentric Form 2")
#         plt.plot(x_values, pointwise_bary2, color='black', label="Barycentric Form 2")
#         plt.legend(loc='best')
#         plt.grid(True)
#         plt.show()
#
#
#         bary_2_sol.append(bary_2)
#
#
#
#
#
#     return bary_2_sol
#
# if __name__ == '__main__':
#     while True:
#         bary_2_sol = task_2_driver()
#
#         user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
#         if user_input != 'y':
#             break
#
#     print("\nThank you for using the Interpolation Driver!")

eps = 2 * np.finfo(float).eps
shift = 1e3 * eps
d = 9
a = 1
b = 3
rho = 2
f = functions_1_to_4.p_1(d, rho)
x_eval = np.linspace(a + shift, b - shift, 100)
exact = f(x_eval)



m = [3, 6, 9, 12]


# x_mesh = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# func_vals = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# bc1 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# bc2 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# newt = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# condition_xny_b1 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# condition_xny_b2 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# condition_xn1_b1 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# condition_xn1_b2 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# Lambda_n_b1 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# Lambda_n_b2 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# relative_error_b2 = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# relative_error_newt_inc = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# relative_error_newt_dec = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
# relative_error_newt_leja = {
#     'uniform': [],
#     'chebyshev_first': [],
#     'chebyshev_second': []
# }
#
#
#
# for d in m:
#     """ CREATING MESHES """
#
#     for type in ['uniform', 'chebyshev_first', 'chebyshev_second']:
#         if type == 'uniform':
#             # Single
#             x_mesh_32_u = ifs.chebyshev_points(a, b, d, flag=1, dtype=np.float32)
#             x_mesh_dec_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=1)
#             x_mesh_inc_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=2)
#             x_mesh_leja_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=3)
#
#             x_mesh[type].append(x_mesh_inc_32_u)
#
#             # BARYCENTRIC 1
#             gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_u, f, dtype=np.float32)
#             b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
#                                                                                    x_mesh_32_u, x_eval, func_val_b1_32,
#                                                                                    dtype=np.float32)
#
#             func_vals[type].append(func_val_b1_32)
#             bc1[type].append(b1_32)
#
#             # BARYCENTRIC 2
#             beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_u, f, flag=1, dtype=np.float32)
#             b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_u,
#                                                                                       x_eval, func_val_b2_32_c1,
#                                                                                       dtype=np.float32)
#
#             bc2[type].append(b2_32_c1)
#
#             # NEWTON
#             func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_u, f, dtype=np.float32)
#             n_32 = ifs.horner_interpolation(x_mesh_inc_32_u, x_eval, div_coeff_32, dtype=np.float32)
#
#             newt[type].append(n_32)
#
#             # Double
#             x_mesh_64_u = ifs.chebyshev_points(a, b, d, flag=1, dtype=np.float64)
#             x_mesh_dec_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=1)
#             x_mesh_inc_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=2)
#             x_mesh_leja_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=3)
#
#             # BARYCENTRIC 1
#             gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_u, f, dtype=np.float64)
#             b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_u, x_eval,
#                                                                              func_val_b1_64,
#                                                                              dtype=np.float64)
#
#             # Relative Error
#             num_err_b1 = np.abs(b1_64 - b1_32)
#             denom_err_b1 = np.abs(b1_64)
#             rel_err_b1 = num_err_b1 / denom_err_b1
#
#             # Conditioning
#             cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
#             condition_xny_b1[type].append(cond_xny_64_b1_c1)
#             condition_xn1_b1[type].append(cond_xn1_64)
#             Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))
#
#             # BARYCENTRIC 2
#             beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_u, f, flag=1, dtype=np.float64)
#             b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_u,
#                                                                                       x_eval, func_val_b2_64,
#                                                                                       dtype=np.float64)
#
#             # Relative Error
#             num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
#             denom_err_b2 = np.abs(b2_64_c1)
#             rel_err_b2 = num_err_b2 / denom_err_b2
#             relative_error_b2[type].append(rel_err_b2)
#
#             # Conditioning
#             condition_xny_b2[type].append(cond_xny_64_b2_c1)
#             condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
#             Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))
#
#
#             # NEWTON
#             func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_u, f, dtype=np.float64)
#             n_64 = ifs.horner_interpolation(x_mesh_inc_64_u, x_eval, div_coeff_64, dtype=np.float64)
#
#             # Relative Error
#             num_err_newt = np.abs(n_64 - n_32)
#             denom_err_newt = np.abs(n_64)
#             rel_err_newt = num_err_newt / denom_err_newt
#             relative_error_newt_inc[type].append(rel_err_newt)
#
#         elif type == 'chebyshev_first':
#             ### SINGLE ###
#             x_mesh_32_c1 = ifs.chebyshev_points(a, b, d, flag=2, dtype=np.float32)
#             x_mesh_dec_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=1)
#             x_mesh_inc_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=2)
#             x_mesh_leja_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=3)
#
#             x_mesh[type].append(x_mesh_32_c1)
#
#             # BARYCENTRIC 1
#             gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c1, f, dtype=np.float32)
#             b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
#                                                                                    x_mesh_32_c1, x_eval, func_val_b1_32,
#                                                                                    dtype=np.float32)
#             func_vals[type].append(func_val_b1_32)
#             bc1[type].append(b1_32)
#
#             # BARYCENTRIC 2
#             beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c1, f, flag=2, dtype=np.float32)
#             b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c1,
#                                                                                       x_eval, func_val_b2_32_c1,
#                                                                                       dtype=np.float32)
#
#             bc2[type].append(b2_32_c1)
#
#             # NEWTON
#             func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_c1, f, dtype=np.float32)
#             n_32 = ifs.horner_interpolation(x_mesh_inc_32_c1, x_eval, div_coeff_32, dtype=np.float32)
#
#             newt[type].append(n_32)
#
#             ### DOUBLE ###
#             x_mesh_64_c1 = ifs.chebyshev_points(a, b, d, flag=2, dtype=np.float64)
#             x_mesh_dec_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=1)
#             x_mesh_inc_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=2)
#             x_mesh_leja_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=3)
#
#             # BARYCENTRIC 1
#             gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_c1, f, dtype=np.float64)
#             b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_c1, x_eval,
#                                                                              func_val_b1_64,
#                                                                              dtype=np.float64)
#
#             # Relative Error
#             num_err_b1 = np.abs(b1_64 - b1_32)
#             denom_err_b1 = np.abs(b1_64)
#             rel_err_b1 = num_err_b1 / denom_err_b1
#
#             # Conditioning
#             cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
#             condition_xny_b1[type].append(cond_xny_64_b1_c1)
#             condition_xn1_b1[type].append(cond_xn1_64)
#             Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))
#
#
#             # BARYCENTRIC 2
#             beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_c1, f, flag=2, dtype=np.float64)
#             b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_c1,
#                                                                                       x_eval, func_val_b2_64,
#                                                                                       dtype=np.float64)
#
#             # Relative Error
#             num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
#             denom_err_b2 = np.abs(b2_64_c1)
#             rel_err_b2 = num_err_b2 / denom_err_b2
#             relative_error_b2[type].append(rel_err_b2)
#
#             # Conditioning
#             condition_xny_b2[type].append(cond_xny_64_b2_c1)
#             condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
#             Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))
#
#             # NEWTON
#             func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_c1, f, dtype=np.float64)
#             n_64 = ifs.horner_interpolation(x_mesh_inc_64_c1, x_eval, div_coeff_64, dtype=np.float64)
#
#             # Relative Error
#             num_err_newt = np.abs(n_64 - n_32)
#             denom_err_newt = np.abs(n_64)
#             rel_err_newt = num_err_newt / denom_err_newt
#             relative_error_newt_inc[type].append(rel_err_newt)
#
#
#         else: # type == 'chebyshev_second'
#             ### Single ###
#             x_mesh_32_c2 = ifs.chebyshev_points(a, b, d, flag=3, dtype=np.float32)
#             x_mesh_dec_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=1)
#             x_mesh_inc_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=2)
#             x_mesh_leja_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=3)
#
#             x_mesh[type].append(x_mesh_32_c2)
#
#             # BARYCENTRIC 1
#             gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c2, f, dtype=np.float32)
#             b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
#                                                                                    x_mesh_32_c2, x_eval, func_val_b1_32,
#                                                                                    dtype=np.float32)
#
#             func_vals[type].append(func_val_b1_32)
#             bc1[type].append(b1_32)
#
#             # BARYCENTRIC 2
#             beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c2, f, flag=3, dtype=np.float32)
#             b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c2,
#                                                                                       x_eval, func_val_b2_32_c1,
#                                                                                       dtype=np.float32)
#
#             bc2[type].append(b2_32_c1)
#
#             # NEWTON
#             func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_c2, f, dtype=np.float32)
#             n_32 = ifs.horner_interpolation(x_mesh_inc_32_c2, x_eval, div_coeff_32, dtype=np.float32)
#
#             newt[type].append(n_32)
#
#             ### Double ###
#             x_mesh_64_c2 = ifs.chebyshev_points(a, b, d, flag=3, dtype=np.float64)
#             x_mesh_dec_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=1)
#             x_mesh_inc_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=2)
#             x_mesh_leja_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=3)
#
#             gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_c2, f, dtype=np.float64)
#             b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_c2, x_eval,
#                                                                              func_val_b1_64,
#                                                                              dtype=np.float64)
#
#             # Relative Error
#             num_err_b1 = np.abs(b1_64 - b1_32)
#             denom_err_b1 = np.abs(b1_64)
#             rel_err_b1 = num_err_b1 / denom_err_b1
#
#             # Conditioning
#             cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
#             condition_xny_b1[type].append(cond_xny_64_b1_c1)
#             condition_xn1_b1[type].append(cond_xn1_64)
#             Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))
#
#             # BARYCENTRIC 2
#             beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_c2, f, flag=3, dtype=np.float64)
#             b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_c2,
#                                                                                       x_eval, func_val_b2_64,
#                                                                                       dtype=np.float64)
#
#             # Relative Error
#             num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
#             denom_err_b2 = np.abs(b2_64_c1)
#             rel_err_b2 = num_err_b2 / denom_err_b2
#             relative_error_b2[type].append(rel_err_b2)
#
#             # Conditioning
#             condition_xny_b2[type].append(cond_xny_64_b2_c1)
#             condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
#             Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))
#
#             # NEWTON
#             func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_c2, f, dtype=np.float64)
#             n_64 = ifs.horner_interpolation(x_mesh_inc_64_c2, x_eval, div_coeff_64, dtype=np.float64)
#
#             # Relative Error
#             num_err_newt = np.abs(n_64 - n_32)
#             denom_err_newt = np.abs(n_64)
#             rel_err_newt = num_err_newt / denom_err_newt
#             relative_error_newt_inc[type].append(rel_err_newt)
#
#
#
#
# type = ['uniform', 'chebyshev_first', 'chebyshev_second']
#
# plt.figure(figsize=(18, 6))
# plt.suptitle(f'Interpolation Methods vs. Exact Function for {m[1]} Mesh Points')
# plt.subplot(1, 3, 1)
# plt.plot(x_mesh['uniform'][1], func_vals['uniform'][1], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['uniform'][1], label='Barycentric 1')
# plt.plot(x_eval, bc2['uniform'][1], label='Barycentric 2')
# plt.plot(x_eval, newt['uniform'][1], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Uniform Mesh for m = {m[1]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.subplot(1, 3, 2)
# plt.plot(x_mesh['chebyshev_first'][1], func_vals['chebyshev_first'][1], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['chebyshev_first'][1], label='Barycentric 1')
# plt.plot(x_eval, bc2['chebyshev_first'][1], label='Barycentric 2')
# plt.plot(x_eval, newt['chebyshev_first'][1], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Chebyshev First Kind Mesh for m = {m[1]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.subplot(1, 3, 3)
# plt.plot(x_mesh['chebyshev_second'][1], func_vals['chebyshev_second'][1], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['chebyshev_second'][1], label='Barycentric 1')
# plt.plot(x_eval, bc2['chebyshev_second'][1], label='Barycentric 2')
# plt.plot(x_eval, newt['chebyshev_second'][1], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Chebyshev Second Kind Mesh for m = {m[1]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.show()
#
# plt.figure(figsize=(18, 6))
# plt.suptitle(f'Interpolation Methods vs. Exact Function for {m[3]} Mesh Points')
# plt.subplot(1, 3, 1)
# plt.plot(x_mesh['uniform'][3], func_vals['uniform'][3], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['uniform'][3], label='Barycentric 1')
# plt.plot(x_eval, bc2['uniform'][3], label='Barycentric 2')
# plt.plot(x_eval, newt['uniform'][3], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Uniform Mesh for m = {m[3]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.subplot(1, 3, 2)
# plt.plot(x_mesh['chebyshev_first'][3], func_vals['chebyshev_first'][3], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['chebyshev_first'][3], label='Barycentric 1')
# plt.plot(x_eval, bc2['chebyshev_first'][3], label='Barycentric 2')
# plt.plot(x_eval, newt['chebyshev_first'][3], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Chebyshev First Kind Mesh for m = {m[3]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.subplot(1, 3, 3)
# plt.plot(x_mesh['chebyshev_second'][3], func_vals['chebyshev_second'][3], '*', label='Interpolation Points')
# plt.plot(x_eval, bc1['chebyshev_second'][3], label='Barycentric 1')
# plt.plot(x_eval, bc2['chebyshev_second'][3], label='Barycentric 2')
# plt.plot(x_eval, newt['chebyshev_second'][3], label='Newton')
# plt.plot(x_eval, exact, label='f(x)')
# plt.xlabel('x values')
# plt.ylabel('Function Values')
# plt.title(f'Chebyshev Second Kind Mesh for m = {m[3]}')
# plt.legend(loc='best')
# plt.grid(True)
#
# plt.show()
#
#
# """BARYCENTRIC 1 K(X,N,Y) PLOT"""
# plt.figure(figsize=(18, 6))
# plt.suptitle('Condition Number k(x,n,y) for Barycentric 1 with Different Meshes')
# plt.subplot(1, 3, 1)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b1['uniform'][i], label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Condition Number k(x,n,y)')
# plt.title('Uniform Mesh')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 2)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b1['chebyshev_first'][i], label=f'm = {m[i]}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Chebyshev of the First Kind Mesh')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 3)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b1['chebyshev_second'][i], label=f'm = {m[i]}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Chebyshev of the Second Kind Mesh')
# plt.legend(loc='best')
#
# plt.show()
#
#
# """BARYCENTRIC 2 K(X,N,Y) PLOT"""
# plt.figure(figsize=(18, 6))
# plt.suptitle('Condition Number k(x,n,y) for Barycentric 2 with Different Meshes')
# plt.subplot(1, 3, 1)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b2['uniform'][i], label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Condition Number k(x,n,y)')
# plt.title('Uniform Mesh')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 2)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b2['chebyshev_first'][i], label=f'm = {m[i]}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Chebyshev of the First Kind Mesh')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 3)
# for i in range(len(m)):
#     plt.plot(x_eval, condition_xny_b2['chebyshev_second'][i], label=f'm = {m[i]}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Chebyshev of the Second Kind Mesh')
# plt.legend(loc='best')
#
# plt.show()
#
# """RELATIVE ERROR PLOT FOR BARYCENTRIC 2"""
# plt.figure(figsize=(18, 6))
# plt.suptitle('Relative Error for Barycentric 2 with Different Type and Size Meshes')
# plt.subplot(1, 3, 1)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_b2['uniform'][i], label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Uniform Mesh Relative Error')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 2)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_b2['chebyshev_first'][i], label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Chebyshev of the First Kind Relative Error')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 3)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_b2['chebyshev_second'][i], label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Chebyshev of the Second Kind Relative Error')
# plt.legend(loc='best')
#
# plt.show()
#
# """RELATIVE ERROR PLOT FOR NEWTON"""
# plt.figure(figsize=(18, 6))
# plt.suptitle('Relative Error For Newton Divided Difference Across Different Meshes with Increasing Ordering')
# plt.subplot(1, 3, 1)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_newt_inc['uniform'][i], '*', label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Increasing Uniform Mesh Relative Error')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 2)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_newt_inc['chebyshev_first'][i], '*', label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Increasing Chebyshev First Kind Mesh Relative Error')
# plt.legend(loc='best')
#
# plt.subplot(1, 3, 3)
# plt.yscale('log')
# for i in range(len(m)):
#     plt.plot(x_eval, relative_error_newt_inc['chebyshev_second'][i], '*', label=f'm = {m[i]}')
# plt.xlabel('x values')
# plt.ylabel('Relative Error')
# plt.title('Increasing Chebyshev Second Kind Mesh Relative Error')
# plt.legend(loc='best')
#
# plt.show()

x_mesh = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
x_mesh_inc = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
x_mesh_dec = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
x_mesh_leja = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
func_vals = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
func_vals_inc = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
func_vals_dec = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
func_vals_leja = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
bc1 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
bc2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
newt = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
newt_inc = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
newt_dec = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
newt_leja = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
condition_xny_b1 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
condition_xny_b2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
condition_xn1_b1 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
condition_xn1_b2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
Lambda_n_b1 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
Lambda_n_b2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
relative_error_b2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
relative_error_newt_inc = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
relative_error_newt_dec = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
relative_error_newt_leja = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}



for d in m:
    """ CREATING MESHES """

    for type in ['uniform', 'chebyshev_first', 'chebyshev_second']:
        if type == 'uniform':
            # Single
            x_mesh_32_u = ifs.chebyshev_points(a, b, d, flag=1, dtype=np.float32)
            x_mesh_dec_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=1)
            x_mesh_inc_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=2)
            x_mesh_leja_32_u = ifs.x_mesh_order(x_mesh_32_u, flag=3)

            x_mesh[type].append(x_mesh_32_u)
            x_mesh_inc[type].append(x_mesh_inc_32_u)
            x_mesh_dec[type].append(x_mesh_dec_32_u)
            x_mesh_leja[type].append(x_mesh_leja_32_u)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_u, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_u, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)

            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_u, f, flag=1, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_u,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_32_u, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_32_u, x_eval, div_coeff_32, dtype=np.float32)

            func_val_newt_inc_32, div_coeff_inc_32 = ifs.newton_divdiff(x_mesh_inc_32_u, f, dtype=np.float32)
            n_32_inc = ifs.horner_interpolation(x_mesh_inc_32_u, x_eval, div_coeff_inc_32, dtype=np.float32)

            func_val_newt_dec_32, div_coeff_dec_32 = ifs.newton_divdiff(x_mesh_dec_32_u, f, dtype=np.float32)
            n_32_dec = ifs.horner_interpolation(x_mesh_dec_32_u, x_eval, div_coeff_dec_32, dtype=np.float32)

            func_val_newt_leja_32, div_coeff_leja_32 = ifs.newton_divdiff(x_mesh_leja_32_u, f, dtype=np.float32)
            n_32_leja = ifs.horner_interpolation(x_mesh_leja_32_u, x_eval, div_coeff_leja_32, dtype=np.float32)


            bc1[type].append(b1_32)
            bc2[type].append(b2_32_c1)
            newt[type].append(n_32)
            newt_inc[type].append(n_32_inc)
            newt_dec[type].append(n_32_dec)
            newt_leja[type].append(n_32_leja)

            func_vals[type].append(func_val_b1_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            # Double
            x_mesh_64_u = ifs.chebyshev_points(a, b, d, flag=1, dtype=np.float64)
            x_mesh_dec_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=1)
            x_mesh_inc_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=2)
            x_mesh_leja_64_u = ifs.x_mesh_order(x_mesh_64_u, flag=3)

            # BARYCENTRIC 1
            gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_u, f, dtype=np.float64)
            b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_u, x_eval,
                                                                             func_val_b1_64,
                                                                             dtype=np.float64)

            # Relative Error
            num_err_b1 = np.abs(b1_64 - b1_32)
            denom_err_b1 = np.abs(b1_64)
            rel_err_b1 = num_err_b1 / denom_err_b1

            # Conditioning
            cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
            condition_xny_b1[type].append(cond_xny_64_b1_c1)
            condition_xn1_b2[type].append(cond_xn1_64)
            Lambda_n_b1[type].append(np.max(np.abs(cond_xn1_64)))

            # BARYCENTRIC 2
            beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_u, f, flag=1, dtype=np.float64)
            b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_u,
                                                                                      x_eval, func_val_b2_64,
                                                                                      dtype=np.float64)

            # Relative Error
            num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
            denom_err_b2 = np.abs(b2_64_c1)
            rel_err_b2 = num_err_b2 / denom_err_b2
            relative_error_b2[type].append(rel_err_b2)

            # Conditioning
            condition_xny_b2[type].append(cond_xny_64_b2_c1)
            condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
            Lambda_n_b2[type].append(np.max(np.abs(cond_xn1_64_b2_c1)))


            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_64_u, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_64_u, x_eval, div_coeff_64, dtype=np.float64)

            func_val_newt_inc_64, div_coeff_inc_64 = ifs.newton_divdiff(x_mesh_inc_64_u, f, dtype=np.float64)
            n_64_inc = ifs.horner_interpolation(x_mesh_inc_64_u, x_eval, div_coeff_inc_64, dtype=np.float64)

            func_val_newt_dec_64, div_coeff_dec_64 = ifs.newton_divdiff(x_mesh_dec_64_u, f, dtype=np.float64)
            n_64_dec = ifs.horner_interpolation(x_mesh_dec_64_u, x_eval, div_coeff_dec_64, dtype=np.float64)

            func_val_newt_leja_64, div_coeff_leja_64 = ifs.newton_divdiff(x_mesh_leja_64_u, f, dtype=np.float64)
            n_64_leja = ifs.horner_interpolation(x_mesh_leja_64_u, x_eval, div_coeff_leja_64, dtype=np.float64)

            # Relative Error
            num_err_newt_inc = np.abs(n_64_inc - n_32_inc)
            denom_err_newt_inc = np.abs(n_64_inc)
            rel_err_newt_inc = num_err_newt_inc / denom_err_newt_inc
            relative_error_newt_inc[type].append(rel_err_newt_inc)

            num_err_newt_dec = np.abs(n_64_dec - n_32_dec)
            denom_err_newt_dec = np.abs(n_64_dec)
            rel_err_newt_dec = num_err_newt_dec / denom_err_newt_dec
            relative_error_newt_dec[type].append(rel_err_newt_dec)

            num_err_newt_leja = np.abs(n_64_leja - n_32_leja)
            denom_err_newt_leja = np.abs(n_64_leja)
            rel_err_newt_leja = num_err_newt_leja / denom_err_newt_leja
            relative_error_newt_leja[type].append(rel_err_newt_leja)


        elif type == 'chebyshev_first':
            ### SINGLE ###
            x_mesh_32_c1 = ifs.chebyshev_points(a, b, d, flag=2, dtype=np.float32)
            x_mesh_dec_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=1)
            x_mesh_inc_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=2)
            x_mesh_leja_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=3)

            x_mesh[type].append(x_mesh_32_c1)
            x_mesh_inc[type].append(x_mesh_inc_32_c1)
            x_mesh_dec[type].append(x_mesh_dec_32_c1)
            x_mesh_leja[type].append(x_mesh_leja_32_c1)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c1, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_c1, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)


            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c1, f, flag=2, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c1,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_32_c1, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_32_c1, x_eval, div_coeff_32, dtype=np.float32)

            func_val_newt_inc_32, div_coeff_inc_32 = ifs.newton_divdiff(x_mesh_inc_32_c1, f, dtype=np.float32)
            n_32_inc = ifs.horner_interpolation(x_mesh_inc_32_c1, x_eval, div_coeff_inc_32, dtype=np.float32)

            func_val_newt_dec_32, div_coeff_dec_32 = ifs.newton_divdiff(x_mesh_dec_32_c1, f, dtype=np.float32)
            n_32_dec = ifs.horner_interpolation(x_mesh_dec_32_c1, x_eval, div_coeff_dec_32, dtype=np.float32)

            func_val_newt_leja_32, div_coeff_leja_32 = ifs.newton_divdiff(x_mesh_leja_32_c1, f, dtype=np.float32)
            n_32_leja = ifs.horner_interpolation(x_mesh_leja_32_c1, x_eval, div_coeff_leja_32, dtype=np.float32)

            bc1[type].append(b1_32)
            bc2[type].append(b2_32_c1)
            newt[type].append(n_32)
            newt_inc[type].append(n_32_inc)
            newt_dec[type].append(n_32_dec)
            newt_leja[type].append(n_32_leja)

            func_vals[type].append(func_val_b1_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            ### DOUBLE ###
            x_mesh_64_c1 = ifs.chebyshev_points(a, b, d, flag=2, dtype=np.float64)
            x_mesh_dec_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=1)
            x_mesh_inc_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=2)
            x_mesh_leja_64_c1 = ifs.x_mesh_order(x_mesh_64_c1, flag=3)

            # BARYCENTRIC 1
            gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_c1, f, dtype=np.float64)
            b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_c1, x_eval,
                                                                             func_val_b1_64,
                                                                             dtype=np.float64)

            # Relative Error
            num_err_b1 = np.abs(b1_64 - b1_32)
            denom_err_b1 = np.abs(b1_64)
            rel_err_b1 = num_err_b1 / denom_err_b1

            # Conditioning
            cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
            condition_xny_b1[type].append(cond_xny_64_b1_c1)
            condition_xny_b1[type].append(cond_xn1_64)
            Lambda_n_b1[type].append(np.max(np.abs(cond_xn1_64)))


            # BARYCENTRIC 2
            beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_c1, f, flag=2, dtype=np.float64)
            b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_c1,
                                                                                      x_eval, func_val_b2_64,
                                                                                      dtype=np.float64)

            # Relative Error
            num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
            denom_err_b2 = np.abs(b2_64_c1)
            rel_err_b2 = num_err_b2 / denom_err_b2
            relative_error_b2[type].append(rel_err_b2)

            # Conditioning
            condition_xny_b2[type].append(cond_xny_64_b2_c1)
            condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
            Lambda_n_b2[type].append(np.max(np.abs(cond_xn1_64_b2_c1)))

            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_64_c1, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_64_c1, x_eval, div_coeff_64, dtype=np.float64)

            func_val_newt_inc_64, div_coeff_inc_64 = ifs.newton_divdiff(x_mesh_inc_64_c1, f, dtype=np.float64)
            n_64_inc = ifs.horner_interpolation(x_mesh_inc_64_c1, x_eval, div_coeff_inc_64, dtype=np.float64)

            func_val_newt_dec_64, div_coeff_dec_64 = ifs.newton_divdiff(x_mesh_dec_64_c1, f, dtype=np.float64)
            n_64_dec = ifs.horner_interpolation(x_mesh_dec_64_c1, x_eval, div_coeff_dec_64, dtype=np.float64)

            func_val_newt_leja_64, div_coeff_leja_64 = ifs.newton_divdiff(x_mesh_leja_64_c1, f, dtype=np.float64)
            n_64_leja = ifs.horner_interpolation(x_mesh_leja_64_c1, x_eval, div_coeff_leja_64, dtype=np.float64)

            # Relative Error
            num_err_newt_inc = np.abs(n_64_inc - n_32_inc)
            denom_err_newt_inc = np.abs(n_64_inc)
            rel_err_newt_inc = num_err_newt_inc / denom_err_newt_inc
            relative_error_newt_inc[type].append(rel_err_newt_inc)

            num_err_newt_dec = np.abs(n_64_dec - n_32_dec)
            denom_err_newt_dec = np.abs(n_64_dec)
            rel_err_newt_dec = num_err_newt_dec / denom_err_newt_dec
            relative_error_newt_dec[type].append(rel_err_newt_dec)

            num_err_newt_leja = np.abs(n_64_leja - n_32_leja)
            denom_err_newt_leja = np.abs(n_64_leja)
            rel_err_newt_leja = num_err_newt_leja / denom_err_newt_leja
            relative_error_newt_leja[type].append(rel_err_newt_leja)


        elif type == 'chebyshev_second': # type == 'chebyshev_second'
            ### Single ###
            x_mesh_32_c2 = ifs.chebyshev_points(a, b, d, flag=3, dtype=np.float32)
            x_mesh_dec_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=1)
            x_mesh_inc_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=2)
            x_mesh_leja_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=3)

            x_mesh[type].append(x_mesh_32_c2)
            x_mesh_inc[type].append(x_mesh_inc_32_c2)
            x_mesh_dec[type].append(x_mesh_dec_32_c2)
            x_mesh_leja[type].append(x_mesh_leja_32_c2)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c2, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_c2, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)

            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c2, f, flag=3, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c2,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_32_c2, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_32_c2, x_eval, div_coeff_32, dtype=np.float32)

            func_val_newt_inc_32, div_coeff_inc_32 = ifs.newton_divdiff(x_mesh_inc_32_c2, f, dtype=np.float32)
            n_32_inc = ifs.horner_interpolation(x_mesh_inc_32_c2, x_eval, div_coeff_inc_32, dtype=np.float32)

            func_val_newt_dec_32, div_coeff_dec_32 = ifs.newton_divdiff(x_mesh_dec_32_c2, f, dtype=np.float32)
            n_32_dec = ifs.horner_interpolation(x_mesh_dec_32_c2, x_eval, div_coeff_dec_32, dtype=np.float32)

            func_val_newt_leja_32, div_coeff_leja_32 = ifs.newton_divdiff(x_mesh_leja_32_c2, f, dtype=np.float32)
            n_32_leja = ifs.horner_interpolation(x_mesh_leja_32_c2, x_eval, div_coeff_leja_32, dtype=np.float32)

            bc1[type].append(b1_32)
            bc2[type].append(b2_32_c1)
            newt[type].append(n_32)
            newt_inc[type].append(n_32_inc)
            newt_dec[type].append(n_32_dec)
            newt_leja[type].append(n_32_leja)

            func_vals[type].append(func_val_b1_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            ### Double ###
            x_mesh_64_c2 = ifs.chebyshev_points(-1, 1, d, flag=3, dtype=np.float64)
            x_mesh_dec_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=1)
            x_mesh_inc_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=2)
            x_mesh_leja_64_c2 = ifs.x_mesh_order(x_mesh_64_c2, flag=3)

            gamma_vec_64, func_val_b1_64 = ifs.coef_gamma(x_mesh_64_c2, f, dtype=np.float64)
            b1_64, cond_xn1_64, cond_numer_xny_64 = ifs.bary_1_interpolation(gamma_vec_64, x_mesh_64_c2, x_eval,
                                                                             func_val_b1_64,
                                                                             dtype=np.float64)

            # Relative Error
            num_err_b1 = np.abs(b1_64 - b1_32)
            denom_err_b1 = np.abs(b1_64)
            rel_err_b1 = num_err_b1 / denom_err_b1

            # Conditioning
            cond_xny_64_b1_c1 = cond_numer_xny_64 / denom_err_b1
            condition_xny_b1[type].append(cond_xny_64_b1_c1)
            condition_xn1_b1[type].append(cond_xn1_64)
            Lambda_n_b1[type].append(np.max(np.abs(cond_xn1_64)))

            # BARYCENTRIC 2
            beta_vec_64_c1, func_val_b2_64 = ifs.coef_beta(x_mesh_64_c2, f, flag=3, dtype=np.float64)
            b2_64_c1, cond_xn1_64_b2_c1, cond_xny_64_b2_c1 = ifs.bary_2_interpolation(beta_vec_64_c1, x_mesh_64_c2,
                                                                                      x_eval, func_val_b2_64,
                                                                                      dtype=np.float64)

            # Relative Error
            num_err_b2 = np.abs(b2_64_c1 - b2_32_c1)
            denom_err_b2 = np.abs(b2_64_c1)
            rel_err_b2 = num_err_b2 / denom_err_b2
            relative_error_b2[type].append(rel_err_b2)

            # Conditioning
            condition_xny_b2[type].append(cond_xny_64_b2_c1)
            condition_xn1_b2[type].append(cond_xn1_64_b2_c1)
            Lambda_n_b2[type].append(np.max(np.abs(cond_xn1_64_b2_c1)))

            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_64_c2, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_64_c2, x_eval, div_coeff_64, dtype=np.float64)

            func_val_newt_inc_64, div_coeff_inc_64 = ifs.newton_divdiff(x_mesh_inc_64_c2, f, dtype=np.float64)
            n_64_inc = ifs.horner_interpolation(x_mesh_inc_64_c2, x_eval, div_coeff_inc_64, dtype=np.float64)

            func_val_newt_dec_64, div_coeff_dec_64 = ifs.newton_divdiff(x_mesh_dec_64_c2, f, dtype=np.float64)
            n_64_dec = ifs.horner_interpolation(x_mesh_dec_64_c2, x_eval, div_coeff_dec_64, dtype=np.float64)

            func_val_newt_leja_64, div_coeff_leja_64 = ifs.newton_divdiff(x_mesh_leja_64_c2, f, dtype=np.float64)
            n_64_leja = ifs.horner_interpolation(x_mesh_leja_64_c2, x_eval, div_coeff_leja_64, dtype=np.float64)

            # Relative Error
            num_err_newt_inc = np.abs(n_64_inc - n_32_inc)
            denom_err_newt_inc = np.abs(n_64_inc)
            rel_err_newt_inc = num_err_newt_inc / denom_err_newt_inc
            relative_error_newt_inc[type].append(rel_err_newt_inc)

            num_err_newt_dec = np.abs(n_64_dec - n_32_dec)
            denom_err_newt_dec = np.abs(n_64_dec)
            rel_err_newt_dec = num_err_newt_dec / denom_err_newt_dec
            relative_error_newt_dec[type].append(rel_err_newt_dec)

            num_err_newt_leja = np.abs(n_64_leja - n_32_leja)
            denom_err_newt_leja = np.abs(n_64_leja)
            rel_err_newt_leja = num_err_newt_leja / denom_err_newt_leja
            relative_error_newt_leja[type].append(rel_err_newt_leja)




type = ['uniform', 'chebyshev_first', 'chebyshev_second']

plt.figure(figsize=(18, 6))
plt.suptitle(f'Interpolation Methods vs. Exact Function for {m[1]} Mesh Points')
plt.subplot(1, 3, 1)
plt.plot(x_mesh['uniform'][1], func_vals['uniform'][1], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['uniform'][1], label='Barycentric 1')
plt.plot(x_eval, bc2['uniform'][1], label='Barycentric 2')
plt.plot(x_eval, newt['uniform'][1], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Uniform Mesh for m = {m[1]}')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_mesh['chebyshev_first'][1], func_vals['chebyshev_first'][1], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['chebyshev_first'][1], label='Barycentric 1')
plt.plot(x_eval, bc2['chebyshev_first'][1], label='Barycentric 2')
plt.plot(x_eval, newt['chebyshev_first'][1], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev First Kind Mesh for m = {m[1]}')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_mesh['chebyshev_second'][1], func_vals['chebyshev_second'][1], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['chebyshev_second'][1], label='Barycentric 1')
plt.plot(x_eval, bc2['chebyshev_second'][1], label='Barycentric 2')
plt.plot(x_eval, newt['chebyshev_second'][1], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev Second Kind Mesh for m = {m[1]}')
plt.legend(loc='best')
plt.grid(True)

plt.show()

plt.figure(figsize=(18, 6))
plt.suptitle(f'Interpolation Methods vs. Exact Function for {m[3]} Mesh Points')
plt.subplot(1, 3, 1)
plt.plot(x_mesh['uniform'][3], func_vals['uniform'][3], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['uniform'][3], label='Barycentric 1')
plt.plot(x_eval, bc2['uniform'][3], label='Barycentric 2')
plt.plot(x_eval, newt['uniform'][3], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Uniform Mesh for m = {m[3]}')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_mesh['chebyshev_first'][3], func_vals['chebyshev_first'][3], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['chebyshev_first'][3], label='Barycentric 1')
plt.plot(x_eval, bc2['chebyshev_first'][3], label='Barycentric 2')
plt.plot(x_eval, newt['chebyshev_first'][3], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev First Kind Mesh for m = {m[3]}')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_mesh['chebyshev_second'][3], func_vals['chebyshev_second'][3], '*', label='Interpolation Points')
plt.plot(x_eval, bc1['chebyshev_second'][3], label='Barycentric 1')
plt.plot(x_eval, bc2['chebyshev_second'][3], label='Barycentric 2')
plt.plot(x_eval, newt['chebyshev_second'][3], label='Newton')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev Second Kind Mesh for m = {m[3]}')
plt.legend(loc='best')
plt.grid(True)

plt.show()

"""BARYCENTRIC 1 K(X,N,Y) PLOT"""
plt.figure(figsize=(18, 6))
plt.suptitle('Condition Number k(x,n,y) for Barycentric 1 with Different Meshes')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b1['uniform'][i], label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Uniform Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b1['chebyshev_first'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Chebyshev of the First Kind Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b1['chebyshev_second'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Chebyshev of the Second Kind Mesh')
plt.legend(loc='best')

plt.show()


"""BARYCENTRIC 2 K(X,N,Y) PLOT"""
plt.figure(figsize=(18, 6))
plt.suptitle('Condition Number k(x,n,y) for Barycentric 2 with Different Meshes')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b2['uniform'][i], label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Uniform Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b2['chebyshev_first'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Chebyshev of the First Kind Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b2['chebyshev_second'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('Condition Number k(x,n,y)')
plt.title('Chebyshev of the Second Kind Mesh')
plt.legend(loc='best')

plt.show()

"""RELATIVE ERROR PLOT FOR BARYCENTRIC 2"""
plt.figure(figsize=(18, 6))
plt.suptitle('Relative Error for Barycentric 2 with Different Type and Size Meshes')
plt.subplot(1, 3, 1)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['uniform'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['chebyshev_first'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev of the First Kind Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['chebyshev_second'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev of the Second Kind Relative Error')
plt.legend(loc='best')

plt.show()

"""RELATIVE ERROR PLOT FOR NEWTON"""
plt.figure(figsize=(18, 6))
plt.suptitle('Relative Error For Newton Divided Difference Across Different Meshes with Increasing Ordering')
plt.subplot(1, 3, 1)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_inc['uniform'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_inc['chebyshev_first'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Chebyshev First Kind Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_inc['chebyshev_second'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Chebyshev Second Kind Mesh Relative Error')
plt.legend(loc='best')

plt.show()

plt.figure(figsize=(18, 6))
plt.suptitle('Relative Error For Newton Divided Difference Across Different Meshes with Decreasing Ordering')
plt.subplot(1, 3, 1)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_dec['uniform'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_dec['chebyshev_first'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev First Kind Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_dec['chebyshev_second'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev Second Kind Mesh Relative Error')
plt.legend(loc='best')

plt.show()

plt.figure(figsize=(18, 6))
plt.suptitle('Relative Error For Newton Divided Difference Across Different Meshes with Leja Ordering')
plt.subplot(1, 3, 1)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_leja['uniform'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_leja['chebyshev_first'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev First Kind Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_leja['chebyshev_second'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev Second Kind Mesh Relative Error')
plt.legend(loc='best')

plt.show()