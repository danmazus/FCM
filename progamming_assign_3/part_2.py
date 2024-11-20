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
                                                                           2: Forward Gauss-Seidel,
                                                                           3: Backward Gauss-Seidel,
                                                                           4: Symmetric Gauss-Seidel)

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

    # Jacobi Method
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

    # Forward Gauss-Seidel
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

    # Backward Gauss-Seidel
    elif flag == 3:
        rel_err_list = []
        iter_num = 0
        pre_cond = D - U

        while iter_num < max_iter:
            rel_err = (vec_2_norm_np(x - x_true)) / (vec_2_norm_np(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Solve Upper Triangular for P^(-1) * r_k
            z = solve_Ux_np(pre_cond, r)

            # Compute next x
            x_next = x + z

            # Compute next residual
            r_next = b - np.dot(A, x_next)

            # Update Values
            r = r_next
            x = x_next
            iter_num += 1

        return x, iter_num, rel_err_list

    # Symmetric Gauss-Seidel
    else: # STILL WORKING ON THIS FUNCTION DO NOT USE
        '''Symmetric Gauss-Seidel'''
        rel_err_list = []
        iter_num = 0
        Lower = D - L
        Upper = D - U

        while iter_num < max_iter:
            # Compute Relative error ||x_k - x|| / ||x||
            rel_err = (vec_2_norm_np(x - x_true)) / (vec_2_norm_np(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # First the Lower Solve is computed first (D - L)^(-1) * r_k = z_1
            z_1 = solve_Lb_np(Lower, r)

            # Scale of the diagonal D * (D - L)^(-1) * r_k = z_2
            z_2 = D * z_1

            # Upper Solve is finally computed (D - U)^(-1) * z_2 = z_3
            z_3 = solve_Ux_np(Upper, z_2)

            # Computing the next x
            x_next = x + z_3

            # Update the residual
            r_next = b - np.dot(A, x_next)

            # Correcting variables for next iteration
            x = x_next
            r = r_next
            iter_num += 1

        return x, iter_num, rel_err_list

def get_user_inputs():
    """
    Get problem parameters from user input.

    Prompts user for:
    - Matrix dimensions (n rows x n columns)
    - Matrix to be passed (A0 through A9)
    - Range for random values for solution vectors (smin, smax)
    - Number of initial guess vectors (g)
    - Range of for random values for initial guess vectors (ig_min_value, ig_max_value)
    - Tolerance level for convergence to happen (tol)
    - Maximum number of iterations before stopping (max_iter)
    - Debug output preference
    """

    print("\nJacobi, Gauss-Seidel, Symmetric Gauss-Seidel Construction")
    print("----------------------------------")

    while True:
        try:
            n = int(input("Enter dimensions for matrices and vectors (n) [default=10]: ") or "10")

            print("\nChoose which matrix to use:")
            print("1. Matrix A_0")
            print("2. Matrix A_1")
            print("3. Matrix A_2")
            print("4. Matrix A_3")
            print("5. Matrix A_4")
            print("6. Matrix A_5")
            print("7. Matrix A_6")
            print("8. Matrix A_7")
            print("9. Matrix A_8")
            problem_type = int(input("Enter Matrix [default=1 (Matrix A_0)]: ") or "1")

            if problem_type not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                print("Error: Invalid problem type")
                continue

            # New inputs for random reflector range
            print("\nSet range for random values of Solution Vectors:")
            smin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
            smax = float(input("Enter maximum value (default=10.0): ") or "10.0")

            if smin >= smax:
                print("Error: Minimum value must be less than maximum value")
                continue

            print("\nChoose How Many Initial Guess Vectors:")
            g = int(input("Enter number of Initial Guess Vectors (default=10): ") or "10")

            print("\nSet range of values for Initial Guess Vectors:")
            ig_value_min = float(input("Enter minimum value (default=0.0): ") or "0.0")
            ig_value_max = float(input("Enter maximum value (default=0.0): ") or "0.0")

            print("\nSet Tolerance Level for Convergence to Hit:")
            tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")

            print("\nSet Maximum Number of Iterations to be Ran:")
            max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")


            debug = input("\nEnable debug output? (y/n) [default=n]: ").lower().startswith('y')

            return n, problem_type, smin, smax, g, ig_value_min, ig_value_max, tol, max_iter, debug

        except ValueError:
            print("Error: Please enter valid numbers")

def part_2_driver():
    # Getting user inputs
    inputs = get_user_inputs()

    n, problem_type, smin, smax, g, ig_value_min, ig_value_max, tol, max_iter, debug = inputs

    # Sets random seed for reproducibility
    #np.random.seed(42)

