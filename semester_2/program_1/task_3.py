import numpy as np
import matplotlib.pyplot as plt
import interpolate_functions as ifs
import functions_1_to_4

"""
This task will perform the subtasks for the f_2 function:
    1.  The interpolating problem that the given polynomial solves on the uniform mesh points and Chebyshev
        points of the first and second kind, i.e., y_i = f(x_i), 0 <= i <= m where m is d for f_2(x). Also,
        include some tests that use more points than necessary to reproduce the polynomial f_2(x). For f_2(x)
        choose at least two different degrees that are greater than 20 for each mesh type.
    2.  For each of the degrees used for the uniform and Chebyshev meshes, determine the conditioning by
        evaluating k(x,n,y( and k(x,n,1) for a <= x <= b and summarize them appropriately using \Lambda_n
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

eps = 2 * np.finfo(float).eps
shift = 1e3 * eps
d = 9
a = 1
b = 3
f = functions_1_to_4.p_2(d)
x_eval = np.linspace(a, b, 1000)
exact = f(x_eval)



m = [3, 6, 9, 12]


x_mesh = {
    'uniform': [],
    'chebyshev_first': [],
    'chebyshev_second': []
}
func_vals = {
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

            x_mesh[type].append(x_mesh_inc_32_u)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_u, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_u, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)

            func_vals[type].append(func_val_b1_32)
            bc1[type].append(b1_32)

            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_u, f, flag=1, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_u,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            bc2[type].append(b2_32_c1)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_u, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_inc_32_u, x_eval, div_coeff_32, dtype=np.float32)

            newt[type].append(n_32)

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
            condition_xn1_b1[type].append(cond_xn1_64)
            Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))

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
            Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))


            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_u, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_inc_64_u, x_eval, div_coeff_64, dtype=np.float64)

            # Relative Error
            num_err_newt = np.abs(n_64 - n_32)
            denom_err_newt = np.abs(n_64)
            rel_err_newt = num_err_newt / denom_err_newt
            relative_error_newt_inc[type].append(rel_err_newt)

        elif type == 'chebyshev_first':
            ### SINGLE ###
            x_mesh_32_c1 = ifs.chebyshev_points(a, b, d, flag=2, dtype=np.float32)
            x_mesh_dec_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=1)
            x_mesh_inc_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=2)
            x_mesh_leja_32_c1 = ifs.x_mesh_order(x_mesh_32_c1, flag=3)

            x_mesh[type].append(x_mesh_32_c1)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c1, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_c1, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)
            func_vals[type].append(func_val_b1_32)
            bc1[type].append(b1_32)

            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c1, f, flag=2, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c1,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            bc2[type].append(b2_32_c1)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_c1, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_inc_32_c1, x_eval, div_coeff_32, dtype=np.float32)

            newt[type].append(n_32)

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
            condition_xn1_b1[type].append(cond_xn1_64)
            Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))


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
            Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))

            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_c1, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_inc_64_c1, x_eval, div_coeff_64, dtype=np.float64)

            # Relative Error
            num_err_newt = np.abs(n_64 - n_32)
            denom_err_newt = np.abs(n_64)
            rel_err_newt = num_err_newt / denom_err_newt
            relative_error_newt_inc[type].append(rel_err_newt)


        else: # type == 'chebyshev_second'
            ### Single ###
            x_mesh_32_c2 = ifs.chebyshev_points(a, b, d, flag=3, dtype=np.float32)
            x_mesh_dec_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=1)
            x_mesh_inc_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=2)
            x_mesh_leja_32_c2 = ifs.x_mesh_order(x_mesh_32_c2, flag=3)

            x_mesh[type].append(x_mesh_32_c2)

            # BARYCENTRIC 1
            gamma_vec_32, func_val_b1_32 = ifs.coef_gamma(x_mesh_32_c2, f, dtype=np.float32)
            b1_32, cond_xn1_32_b1, cond_numer_xny_32_b1 = ifs.bary_1_interpolation(gamma_vec_32,
                                                                                   x_mesh_32_c2, x_eval, func_val_b1_32,
                                                                                   dtype=np.float32)

            func_vals[type].append(func_val_b1_32)
            bc1[type].append(b1_32)

            # BARYCENTRIC 2
            beta_vec_32_c1, func_val_b2_32_c1 = ifs.coef_beta(x_mesh_32_c2, f, flag=3, dtype=np.float32)
            b2_32_c1, cond_xn1_32_b2_c1, cond_xny_32_b2_c1 = ifs.bary_2_interpolation(beta_vec_32_c1, x_mesh_32_c2,
                                                                                      x_eval, func_val_b2_32_c1,
                                                                                      dtype=np.float32)

            bc2[type].append(b2_32_c1)

            # NEWTON
            func_val_newt_32, div_coeff_32 = ifs.newton_divdiff(x_mesh_inc_32_c2, f, dtype=np.float32)
            n_32 = ifs.horner_interpolation(x_mesh_inc_32_c2, x_eval, div_coeff_32, dtype=np.float32)

            newt[type].append(n_32)

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
            Lambda_n_b1[type].append(np.nanmax(cond_xn1_64))

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
            Lambda_n_b2[type].append(np.max(cond_xn1_64_b2_c1))

            # NEWTON
            func_val_newt_64, div_coeff_64 = ifs.newton_divdiff(x_mesh_inc_64_c2, f, dtype=np.float64)
            n_64 = ifs.horner_interpolation(x_mesh_inc_64_c2, x_eval, div_coeff_64, dtype=np.float64)

            # Relative Error
            num_err_newt = np.abs(n_64 - n_32)
            denom_err_newt = np.abs(n_64)
            rel_err_newt = num_err_newt / denom_err_newt
            relative_error_newt_inc[type].append(rel_err_newt)




type = ['uniform', 'chebyshev_first', 'chebyshev_second']

print(x_mesh)

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
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['uniform'][i], label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['chebyshev_first'][i], label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Chebyshev of the First Kind Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_b2['chebyshev_second'][i], label=f'm = {m[i]}')
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
    plt.plot(x_eval, relative_error_newt_inc['uniform'][i], '*', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Uniform Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 2)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_inc['chebyshev_first'][i], '*', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Chebyshev First Kind Mesh Relative Error')
plt.legend(loc='best')

plt.subplot(1, 3, 3)
plt.yscale('log')
for i in range(len(m)):
    plt.plot(x_eval, relative_error_newt_inc['chebyshev_second'][i], '*', label=f'm = {m[i]}')
plt.xlabel('x values')
plt.ylabel('Relative Error')
plt.title('Increasing Chebyshev Second Kind Mesh Relative Error')
plt.legend(loc='best')

plt.show()