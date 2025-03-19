import numpy as np
import matplotlib.pyplot as plt
from semester_2 import interpolate_functions as ifs
import functions_1_to_4
import pandas as pd


"""CREATION OF ALL DICTIONARIES USED FOR STORAGE OF VALUES AND FUNCTION"""

eps = 2 * np.finfo(float).eps
shift = 1e3 * eps
a = -1
b = 1
f = functions_1_to_4.p_4(2)
x_eval = np.linspace(a + shift, b - shift, 100)
exact = f(x_eval)

m = [6, 12, 20, 29]

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
pw_error_b1 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
pw_error_b2 = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
pw_error_newt = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
pw_error_newt_inc = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
pw_error_newt_dec = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
pw_error_newt_leja = {
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

            func_vals[type].append(func_val_newt_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            # Pointwise Error
            pw_err_b1 = np.max(np.abs(exact - b1_32))
            pw_err_b2 = np.max(np.abs(exact - b2_32_c1))
            pw_err_newt = np.max(np.abs(exact - n_32))
            pw_err_newt_inc = np.max(np.abs(exact - n_32_inc))
            pw_err_newt_dec = np.max(np.abs(exact - n_32_dec))
            pw_err_newt_leja = np.max(np.abs(exact - n_32_leja))

            pw_error_b1[type].append(pw_err_b1)
            pw_error_b2[type].append(pw_err_b2)
            pw_error_newt[type].append(pw_err_newt)
            pw_error_newt_inc[type].append(pw_err_newt_inc)
            pw_error_newt_dec[type].append(pw_err_newt_dec)
            pw_error_newt_leja[type].append(pw_err_newt_leja)


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
            Lambda_n_b2[type].append(np.nanmax(np.abs(cond_xn1_64_b2_c1)))


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

            func_vals[type].append(func_val_newt_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            # Pointwise Error
            pw_err_b1 = np.max(np.abs(exact - b1_32))
            pw_err_b2 = np.max(np.abs(exact - b2_32_c1))
            pw_err_newt = np.max(np.abs(exact - n_32))
            pw_err_newt_inc = np.max(np.abs(exact - n_32_inc))
            pw_err_newt_dec = np.max(np.abs(exact - n_32_dec))
            pw_err_newt_leja = np.max(np.abs(exact - n_32_leja))

            pw_error_b1[type].append(pw_err_b1)
            pw_error_b2[type].append(pw_err_b2)
            pw_error_newt[type].append(pw_err_newt)
            pw_error_newt_inc[type].append(pw_err_newt_inc)
            pw_error_newt_dec[type].append(pw_err_newt_dec)
            pw_error_newt_leja[type].append(pw_err_newt_leja)

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
            Lambda_n_b1[type].append(np.nanmax(np.abs(cond_xn1_64)))


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
            Lambda_n_b2[type].append(np.nanmax(np.abs(cond_xn1_64_b2_c1)))

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

            func_vals[type].append(func_val_newt_32)
            func_vals_inc[type].append(func_val_newt_inc_32)
            func_vals_dec[type].append(func_val_newt_dec_32)
            func_vals_leja[type].append(func_val_newt_leja_32)

            # Pointwise Error
            pw_err_b1 = np.max(np.abs(exact - b1_32))
            pw_err_b2 = np.max(np.abs(exact - b2_32_c1))
            pw_err_newt = np.max(np.abs(exact - n_32))
            pw_err_newt_inc = np.max(np.abs(exact - n_32_inc))
            pw_err_newt_dec = np.max(np.abs(exact - n_32_dec))
            pw_err_newt_leja = np.max(np.abs(exact - n_32_leja))

            pw_error_b1[type].append(pw_err_b1)
            pw_error_b2[type].append(pw_err_b2)
            pw_error_newt[type].append(pw_err_newt)
            pw_error_newt_inc[type].append(pw_err_newt_inc)
            pw_error_newt_dec[type].append(pw_err_newt_dec)
            pw_error_newt_leja[type].append(pw_err_newt_leja)

            ### Double ###
            x_mesh_64_c2 = ifs.chebyshev_points(a, b, d, flag=3, dtype=np.float64)
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
            Lambda_n_b2[type].append(np.nanmax(np.abs(cond_xn1_64_b2_c1)))

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

"""INTERPOLATION PLOTS"""
plt.figure(figsize=(18, 6))
plt.suptitle(f'Newton Method vs. f(x) for Given Degree')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, newt_inc['uniform'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Newton Uniform Mesh')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, newt_inc['chebyshev_first'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Newton Chebyshev First Kind')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, newt_inc['chebyshev_second'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Newton Chebyshev Second Kind')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.figure(figsize=(18,6))
plt.suptitle(f'Barycentric 1 Method vs. f(x) for Given Degree')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, bc1['uniform'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Uniform Mesh for Barycentric 1')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, bc1['chebyshev_first'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev First Kind for Barycentric 1')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, bc1['chebyshev_second'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev Second Kind for Barycentric 1')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.figure(figsize=(18,6))
plt.suptitle(f'Barycentric 2 Method vs. f(x) for Given Degree')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, bc2['uniform'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values for Barycentric 2')
plt.title(f'Uniform Mesh')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, bc2['chebyshev_first'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev First Kind for Barycentric 2')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, bc2['chebyshev_second'][i], label=f'm = {m[i]}')
plt.plot(x_eval, exact, label='f(x)')
plt.xlabel('x values')
plt.ylabel('Function Values')
plt.title(f'Chebyshev Second Kind for Barycentric 2')
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
plt.ylabel('y')
plt.title('Chebyshev of the First Kind Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b1['chebyshev_second'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('y')
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
plt.ylabel('y')
plt.title('Chebyshev of the First Kind Mesh')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
for i in range(len(m)):
    plt.plot(x_eval, condition_xny_b2['chebyshev_second'][i], label=f'm = {m[i]}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Chebyshev of the Second Kind Mesh')
plt.legend(loc='best')

plt.show()

"""RELATIVE ERROR PLOT FOR BARYCENTRIC 2"""
plt.figure(figsize=(18, 6))
plt.suptitle('Relative Error for Barycentric 2 with Different Type and Size Meshes')
plt.subplot(1, 3, 1)
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['uniform'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['chebyshev_first'][i], 'x', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev of the First Kind Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
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

### PRINTING H_n AND TABLE OF LAMBDAS ###

mean_u_b1 = np.mean(Lambda_n_b1['uniform'])
mean_u_b2 = np.mean(Lambda_n_b2['uniform'])
mean_c1_b1 = np.mean(Lambda_n_b1['chebyshev_first'])
mean_c1_b2 = np.mean(Lambda_n_b2['chebyshev_first'])
mean_c2_b1 = np.mean(Lambda_n_b1['chebyshev_second'])
mean_c2_b2 = np.mean(Lambda_n_b2['chebyshev_second'])

H_n_b1 = pd.DataFrame({
    'Uniform': [mean_u_b1, mean_u_b2],
    'Chebyshev First Kind': [mean_c1_b1, mean_c1_b2],
    'Chebyshev Second Kind': [mean_c2_b1, mean_c2_b2]
}, index=['Barycentric 1', 'Barycentric 2'])
print(H_n_b1)

lamb_tab_b1 = pd.DataFrame(Lambda_n_b1)
lamb_tab_b2 = pd.DataFrame(Lambda_n_b2)

lamb_tab_b1['Mesh Size'] = m
lamb_tab_b2['Mesh Size'] = m

print(lamb_tab_b1)
print(lamb_tab_b2)

u = np.finfo(np.float32).eps

n = m[3]

bound = ((((3 * n) + 4) * u) * condition_xny_b2['uniform'][3]) + ((((3 * n) + 2) * u) * Lambda_n_b2['uniform'][3])

"""RELATIVE ERROR PLOT WITH BOUND FOR CHEBYSHEV FIRST FOR BARYCENTRIC 2"""
plt.figure(figsize=(10, 6))
plt.yscale('log')
plt.plot(x_eval, relative_error_b2['chebyshev_first'][3], 'x', label=f'm = {m[3]}')
plt.plot(x_eval, bound, label='bound')
plt.xlabel('x values')
plt.ylabel('Log Relative Error')
plt.title('Chebyshev Second Kind Points Relative Error Bound')
plt.legend(loc='best')
plt.show()

### POINTWISE ERROR ###
pw_error_b1_df = pd.DataFrame(pw_error_b1)
pw_error_b2_df = pd.DataFrame(pw_error_b2)
pw_error_newt_df = pd.DataFrame(pw_error_newt)

pw_error_b1_df['Method'] = 'Barycentric 1'
pw_error_b2_df['Method'] = 'Barycentric 2'
pw_error_newt_df['Method'] = 'Newton'

pw_error_b1_df['m'] = m
pw_error_b2_df['m'] = m
pw_error_newt_df['m'] = m

pw_error_df = pd.concat([pw_error_b1_df, pw_error_b2_df, pw_error_newt_df], axis=0, ignore_index=True)


print(pw_error_df)


# def get_user_inputs():
#     m_low = int(input('Please Enter the Minimum Number of Mesh Points to be Used [default = 5]: ') or 5)
#     m_high = int(input('Please Enter the Maximum Number of Mesh Points to be Used [default = 29]: ') or 29)
#     a = float(input('Please Enter the lower bound on the interval to be interpolated [default = -1.0]: ') or -1.0)
#     b = float(input('Please enter the upper bound on the interval to be interpolated [default = 1.0]: ') or 1.0)
#     num_tests = int(input('Please Enter the number of tests [default = 1]: ') or 1)
#
#     return m_low, m_high, a, b, num_tests
#
# def task_5_driver():
#     inputs = get_user_inputs()
#
#     m_low, m_high, a, b, num_tests = inputs
#
#     eps = 2 * np.finfo(float).eps
#     shift = 1e3 * eps
#
#     x_eval = np.linspace(a + shift, b - shift, 1000)




