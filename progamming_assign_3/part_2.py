import numpy as np
from my_package import vec_2_norm_np, solve_Lb_np, solve_Ux_np


# Stationary Methods Function that includes Jacobi, Gauss-Seidel, Symmetric Gauss-Seidel Methods
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

# User input function
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

    # Defining the dictionary of test matrices
    matrices = {
        "A_0": np.array([3, 7, -1, 7, 4, 1, -1, 1, 2]).reshape(3, 3),
        "A_1": np.array([3, 0, 4, 7, 4, 2, -1, -1, 2]).reshape(3, 3),
        "A_2": np.array([-3, 3, -6, -4, 7, -8, 5, 7, -9]).reshape(3, 3),
        "A_3": np.array([4, 1, 1, 2, -9, 0, 0, -8, -6]).reshape(3, 3),
        "A_4": np.array([7, 6, 9, 4, 5, -4, -7, -3, 8]).reshape(3, 3),
        "A_5": np.array([6, -2, 0, -1, 2, -1, 0, -6 / 5, 1]).reshape(3, 3),
        "A_6": np.array([5, -1, 0, -1, 2, -1, 0, -3 / 2, 1]).reshape(3, 3),
        "A_7": np.array([4, -1, 0, 0, 0, 0, 0,
                    -1, 4, -1, 0, 0, 0, 0,
                    0, -1, 4, -1, 0, 0, 0,
                    0, 0, -1, 4, -1, 0, 0,
                    0, 0, 0, -1, 4, -1, 0,
                    0, 0, 0, 0, -1, 4, -1,
                    0, 0, 0, 0, 0, -1, 4]).reshape(7, 7),
        "A_8": np.array([2, -1, 0, 0, 0, 0, 0,
                    -1, 2, -1, 0, 0, 0, 0,
                    0, -1, 2, -1, 0, 0, 0,
                    0, 0, -1, 2, -1, 0, 0,
                    0, 0, 0, -1, 2, -1, 0,
                    0, 0, 0, 0, -1, 2, -1,
                    0, 0, 0, 0, 0, -1, 2]).reshape(7, 7)
    }

    while True:
        try:
            n = int(input("Enter dimensions for vectors (n) [default=10]: ") or "10")

            print("\nChoose which matrix to use:")

            # Dynamically selecting which matrix to use from the dictionary above
            for index, key in enumerate(matrices.keys(), start = 1):
                print(f"{index}. Matrix {key}")
            problem_type = int(input("Enter Matrix [default=1 (Matrix A_0)]: ") or "1")

            # Making sure a correct value was chosen for problem_type
            if problem_type not in range(1, len(matrices) + 1):
                print("Error: Invalid Matrix selection. Please try again.")
                continue

            # Selecting the matrix from the user input, converts the keys to a list and then subtracts 1 from problem type since matrices start at 0
            selected_matrix_key = list(matrices.keys())[problem_type - 1]

            # Selecting the matrix from the key retrieved above
            selected_matrix = matrices[selected_matrix_key]

            # Random values for solution vectors to be used
            print("\nSet range for random values of Solution Vectors:")
            smin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
            smax = float(input("Enter maximum value (default=10.0): ") or "10.0")

            # Checking condition to make sure input is valid
            if smin >= smax:
                print("Error: Minimum value must be less than maximum value")
                continue

            # Choosing how many initial guess vectors will be taken
            print("\nChoose How Many Initial Guess Vectors:")
            g = int(input("Enter number of Initial Guess Vectors (default=10): ") or "10")

            # Random value range for the initial guess vectors
            print("\nSet range of values for Initial Guess Vectors:")
            ig_value_min = float(input("Enter minimum value (default=0.0): ") or "0.0")
            ig_value_max = float(input("Enter maximum value (default=0.0): ") or "0.0")

            # Tolerance level for convergence
            print("\nSet Tolerance Level for Convergence to Hit:")
            tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")

            # Max number of iterations
            print("\nSet Maximum Number of Iterations to be Ran:")
            max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")

            # Selecting which method to run for the selected matrix
            print("\nChoose which method to run:")
            print("1. Jacobi Method")
            print("2. Forward Gauss-Seidel Method")
            print("3. Backward Gauss-Seidel Method")
            print("4. Symmetric Gauss-Seidel Method")
            flag = int(input("Enter Method: "))

            # Ensuring a correct value for method is chosen
            if flag not in [1, 2, 3, 4]:
                print("Error: Invalid method")
                continue

            # Enable debug input
            debug = input("\nEnable debug output? (y/n) [default=n]: ").lower().startswith('y')

            return n, selected_matrix, smin, smax, g, ig_value_min, ig_value_max, tol, max_iter, flag, debug

        except ValueError:
            print("Error: Please enter valid numbers")

# Driver function for the methods
def part_2_driver():
    # Getting user inputs
    inputs = get_user_inputs()

    n, selected_matrix, smin, smax, g, ig_value_min, ig_value_max, tol, max_iter, flag, debug = inputs

    # Sets random seed for reproducibility
    #np.random.seed(42)

    # Setting Initial Conditions
    k = len(selected_matrix)
    x0 = np.array(k)
    x_tilde = np.ones(k)
    b = np.dot(selected_matrix, x_tilde)

    # Jacobi Method
    if flag == 1:
        print("\nUsing Jacobi Method with selected matrix:")
        solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

    # Forward Gauss-Seidel Method
    elif flag == 2:
        print("\nUsing Forward Gauss-Seidel Method with selected matrix:")
        solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

    # Backward Gauss-Seidel Method
    elif flag == 3:
        print("\nUsing Backward Gauss-Seidel Method with selected matrix:")
        solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

    # Symmetric Gauss-Seidel Method
    else: # flag == 4
        print("\nUsing Symmetric Gauss-Seidel Method with selected matrix:")
        solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

    # Printing 1-time results
    print(f"Results")
    print(f"Solution vector is: {solution}")
    print(f"Number of iterations: {iteration}")
    print(f"Relative error: {relative_error}")

    return solution, iteration, relative_error

# Main function
if __name__ == "__main__":
    while True:
        solution, iteration, relative_error = part_2_driver()

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("Thank you for using the Solver!")
