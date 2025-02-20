import numpy as np
import matplotlib.pyplot as plt
import interpolate_functions as ifs
import functions_1_to_4

m = 29
eps = 2 * np.finfo(float).eps
shift = 1e3 * eps
f = functions_1_to_4.p_4(2)


"""EXACT COMPUTATIONS (FLOAT 64)"""
x_mesh_exact = ifs.chebyshev_points(-1, 1, m, flag=2, dtype=np.float64)
x_mesh_inc = ifs.x_mesh_order(x_mesh_exact, flag=2)

gamma_vec_exact, func_val = ifs.coef_gamma(x_mesh_inc, m, f, dtype=np.float64)
beta_vec_exact, func_val_2 = ifs.coef_beta(x_mesh_inc, m, f, flag=2, dtype=np.float64)

x_eval = np.linspace(-1 + shift, 1 - shift, 100)

exact = ifs.bary_1_interpolation(gamma_vec_exact, x_mesh_inc, x_eval, func_val, m, dtype=np.float64)
p_eval_2 = ifs.bary_2_interpolation(beta_vec_exact, x_mesh_inc, x_eval, func_val_2, m, dtype=np.float64)

"""APPROXIMATE COMPUTATIONS (FLOAT 32)"""
x_mesh = ifs.chebyshev_points(-1, 1, m, flag=2, dtype=np.float32)
x_mesh_inc_app = ifs.x_mesh_order(x_mesh, flag=2)

gamma_vec, func_val_app = ifs.coef_gamma(x_mesh_inc_app, m, f, dtype=np.float32)
beta_vec, func_val_2_app = ifs.coef_beta(x_mesh_inc_app, m, f, flag=2,dtype=np.float32)


approx = ifs.bary_1_interpolation(gamma_vec, x_mesh_inc_app, x_eval, func_val_app, m, dtype=np.float32)
approx_2 = ifs.bary_2_interpolation(beta_vec, x_mesh_inc_app, x_eval, func_val_2_app, m, dtype=np.float32)

num_err_b1 = np.abs(exact - approx)
denom_err_b1 = np.abs(exact)

num_err_b2 = np.abs(p_eval_2 - approx_2)
denom_err_b2 = np.abs(p_eval_2)

f_ex = f(x_eval)

rel_err_b1 = num_err_b1 / denom_err_b1
rel_err_b2 = num_err_b2 / denom_err_b2


plt.title("Relative Error")
plt.yscale("log")
plt.plot(x_eval, rel_err_b1, 'x', color='red', label = "Barycentric Form 1")
plt.plot(x_eval, rel_err_b2, 'o', color='black', mfc = "none", label = "Barycentric Form 2")
plt.legend(loc = "best")
plt.grid(True)
plt.show()

plt.title("Interpolation")
plt.plot(x_mesh_inc_app, func_val_2_app, 'x', color='red', label = 'Interpolation Points')
plt.plot(x_eval, approx_2, label='Barycentric Form 2')
plt.plot(x_eval, f_ex, color='black', label='f(x)')
plt.legend(loc = "best")
plt.grid(True)
plt.show()


