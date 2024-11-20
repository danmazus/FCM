import numpy as np
from my_package import vec_2_norm_np, solve_Lb_np, solve_Ux_np


def stationary_method(A, b, x0, x_tilde, tol, max_iter, flag):
    """
        Stationary Methods Function which has parameters consisting of:
            A: Matrix A
            b: Vector b from Ax = b
            x0: Initial guess for x
            x_tilde: Solution vector from Ax = b
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            flag: Flag to indicate which stationary method should be used (1: Jacobi,
                                                                           2: Gauss-Seidel,
                                                                           3: Symmetric Gauss-Seidel)

        Returns:
            x: Solution vector to iteration
            iter_num: Number of iterations for convergence
            rel_err_list: List of Relative Errors during each iteration
    """

    x = x0
    x_true = x_tilde
    r = b - np.dot(A, x)
    D = np.diag(A)
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    if flag == 1:
        """Jacobi"""
        rel_err_list = []
        iter_num = 0
        pre_cond = 1 / D

        while iter_num < max_iter:
            # Computing relative error ||x_k - x_true|| / ||x_true||
            rel_err = (vec_2_norm_np(x - x_true)) / (vec_2_norm_np(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Computing next x term
            x_next = x + pre_cond * r

            # Computing r_(k+1)
            r_next = b - np.dot(A, x_next)

            # Updating Values
            r = r_next
            x = x_next
            iter_num += 1

            return x, iter_num, rel_err_list

    elif flag == 2:
        '''Gauss-Seidel (Forward)'''
        rel_err_list = []
        iter_num = 0
        pre_cond = D - L


        while iter_num < max_iter:
            rel_err = (vec_2_norm_np(x - x_true)) / (vec_2_norm_np(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Lower triangular solve for P^(-1) * r_k
            z = solve_Lb_np(pre_cond, r)

            # Compute the next x term
            x_next = x + z

            # Compute next residual
            r_next = b - np.dot(A, x_next)

            # Update next values
            r = r_next
            x = x_next
            iter_num += 1

            return x, iter_num, rel_err_list


    else: # STILL WORKING ON THIS FUNCTION DO NOT USE
        '''Symmetric Gauss-Seidel'''
        rel_err_list = []
        iter_num = 0
        D_inv = 1 / D
        Lower = D - L
        Upper = D - U

        while iter_num < max_iter:
            # Compute Relative error ||x_k - x|| / ||x||
            rel_err = (vec_2_norm_np(x - x_true)) / (vec_2_norm_np(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            z_1 = solve_Ux_np(Upper, r)
            z_2 = D_inv * r
            z_3 = solve_Lb_np(Lower, r)

