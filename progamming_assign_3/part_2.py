import numpy as np
import matplotlib.pyplot as plt
from my_package import solve_Lb_np, solve_Ux_np, generate_float_1D_vector_np
from scipy.linalg import solve_triangular


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
    r = b - (np.dot(A, x))
    D = np.diag(A)
    L = np.tril(A)
    U = np.triu(A)

    # Jacobi Method
    if flag == 1:
        """Jacobi"""
        rel_err_list = []
        iter_num = 0
        pre_cond = 1 / D

        while iter_num < max_iter:
            # Computing relative error ||x_k - x_true|| / ||x_true||
            rel_err = (np.linalg.norm(x - x_true)) / (np.linalg.norm(x_true))
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
        pre_cond = L    # (D - L)
        true_err_norm = np.linalg.norm(x_true)


        while iter_num < max_iter:
            # Lower triangular solve for P^(-1) * r_k
            z = solve_Lb_np(pre_cond, r)

            # Compute the next x term
            x_next = x + z

            rel_err = (np.linalg.norm(x_next - x_true)) / true_err_norm
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Update next values
            x = x_next
            r = b - np.dot(A, x)
            iter_num += 1

        return x, iter_num, rel_err_list

    # Backward Gauss-Seidel
    elif flag == 3:
        rel_err_list = []
        iter_num = 0
        pre_cond = U

        while iter_num < max_iter:
            # Solve Upper Triangular for P^(-1) * r_k
            z = solve_Ux_np(pre_cond, r)

            # Compute next x
            x_next = x + z

            # Computing the relative error and appending
            rel_err = (np.linalg.norm(x - x_true)) / (np.linalg.norm(x_true))
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Compute next residual
            r_next = b - np.dot(A, x_next)

            # Update Values
            r = r_next
            x = x_next
            iter_num += 1

        return x, iter_num, rel_err_list

    # Symmetric Gauss-Seidel
    else:
        '''Symmetric Gauss-Seidel'''
        rel_err_list = []
        true_err_norm = np.linalg.norm(x_true)
        iter_num = 0
        Lower = L
        Upper = U

        while iter_num < max_iter:
            # First the Lower Solve is computed first (D - L)^(-1) * r_k = z_1
            z_1 = solve_Lb_np(Lower, r)

            # Scale of the diagonal D * (D - L)^(-1) * r_k = z_2
            z_2 = D * z_1

            # Upper Solve is finally computed (D - U)^(-1) * z_2 = z_3
            z_3 = solve_Ux_np(Upper, z_2)

            # Computing the next x
            x = x + z_3

            # Compute Relative error ||x_k - x|| / ||x||
            rel_err = (np.linalg.norm(x - x_true)) / true_err_norm
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Correcting variables for next iteration
            r = b - np.dot(A, x)
            iter_num += 1

        return x, iter_num, rel_err_list

# G matrix computation function
def G_error_matrix(A, k, flag):
    """
    Computes the matrix G which is related to ||Ge_k|| = G^k * ||e_0||

    Parameters:
        A: Given Matrix
        k: size of given matrix
        flag: Flag to indicate which stationary method is being used so corresponding G can be computed

    Returns:
        G: The matrix G relating to ||Ge_k||
    """
    I = np.identity(k)
    D = np.diag(A)
    L = np.tril(A)
    U = np.triu(A)

    if flag == 1:
        D_inv = 1 / D
        B = np.dot(D_inv, A)
        G = I - B

    elif flag == 2:
        B = solve_triangular(L, A, lower=True)
        G = I - B

    elif flag == 3:
        B = solve_triangular(U, A, lower=False)
        G = I - B

    else:
        B_1 = solve_triangular(L, A, lower=True)
        B_2 = np.dot(np.diag(D), B_1)
        B_3 = solve_triangular(U, B_2, lower=False)
        G = I - B_3

    return G

# Spectral Radii Function
def spectral_radi(G):
    """
    Computes the spectral radius of a given matrix

    Parameters:
        G: The given matrix which the spectral radius is wanting to be computed

    Returns:
        spectral: the spectral radius of given matrix
    """
    eigenvalues = np.linalg.eigvals(G)
    spectral = np.max(np.abs(eigenvalues))

    return spectral

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
            #n = int(input("Enter dimensions for vectors (n) [default=10]: ") or "10")

            print("\nChoose which matrix to use:")

            # Dynamically selecting which matrix to use from the dictionary above
            for index, key in enumerate(matrices.keys(), start = 0):
                print(f"{index}. Matrix {key}")
            problem_type = int(input("Enter Matrix (0-8) [default=0 (Matrix A_0)]: ") or "0")

            # Making sure a correct value was chosen for problem_type
            if problem_type < 0 or problem_type > len(matrices) - 1:
                print("Error: Invalid Matrix selection. Please try again.")
                continue

            # Selecting the matrix from the user input, converts the keys to a list and then subtracts 1 from problem type since matrices start at 0
            selected_matrix_key = list(matrices.keys())[problem_type]

            # Selecting the matrix from the key retrieved above
            selected_matrix = matrices[selected_matrix_key]

            # Random values for solution vectors to be used
            print("\nSet range for random values of Solution Vectors:")
            smin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
            smax = float(input("Enter maximum value (default=10.0): ") or "10.0")

            # Checking condition to make sure input is valid
            if smin > smax:
                print("Error: Minimum value must be less than maximum value")
                continue

            # Choosing how many initial guess vectors will be taken
            print("\nChoose How Many Solution Vectors:")
            g = int(input("Enter number of Solution Vectors (default=10): ") or "10")

            print("\nSet range for random values of Initial Guess Vectors:")
            dmin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
            dmax = float(input("Enter maximum value (default=10.0): ") or "10.0")

            if dmin > dmax:
                print("Error: Minimum value must be less than maximum value")
                continue

            print("\nChoose How Many Initial Guess Vectors for each Solution Vector:")
            num_x0 = int(input("Enter number of Initial Guess Vectors (default=10): ") or "10")

            # Tolerance level for convergence
            print("\nSet Tolerance Level for Convergence to Hit:")
            tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")

            # Max number of iterations
            print("\nSet Maximum Number of Iterations to be Ran:")
            max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")

            # Enable debug input
            debug = input("\nEnable debug output? (y/n) (default=n): ").lower().startswith('y')

            ini = input("\nRun initial case for true solution and initial guess vector? (y/n) (default=n): ").lower().startswith('y')
            # If 'n', skip asking for method flag and set defaults
            if not ini:
                # Set default values for these variables
                flag = 1  # Default method, for example Jacobi
                return selected_matrix, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag

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

            return selected_matrix, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag

        except ValueError:
            print("Error: Please enter valid numbers")

# Driver function for the methods
def part_2_driver_one(selected_matrix, smin, smax, g, tol, max_iter, debug, flag):
    # Getting user inputs
    #inputs = get_user_inputs()

    #selected_matrix, smin, smax, g, tol, max_iter, debug, ini, flag = inputs

    # Sets random seed for reproducibility
    #np.random.seed(42)

    # Setting Initial Conditions
    rows, cols = selected_matrix.shape
    k = rows
    x0 = np.zeros(k)

    if debug:
        print(f"Matrix Selected is: \n{selected_matrix}")
        print(f"x0 is: {x0}")

    # Initializing lists to hold values for output
    solution_list = []
    iteration_list = []
    relative_error_list = []
    G_matrix_list = []
    spectral_radius_list = []
    G_matrix_norms_list = []

    # For loop running over multiple solution vectors for same matrix and initial guess vector
    for i in range(g):

        # Setting a new x_tilde (solution vector) time to test over multiple solution vectors for same matrix and same initial guess vector
        x_tilde = generate_float_1D_vector_np(smin, smax, k)
        b = np.dot(selected_matrix, x_tilde)

        # Debug print statements for initial guess, true solution, and b vectors
        if debug:
            print(f"Initial Guess Vector x0 = {x0}")
            print(f"True Solution Vector x_tilde = {x_tilde}")
            print(f"Ax = b is: {b}")

        # Jacobi Method
        if flag == 1:
            print(f"\nUsing Jacobi Method with selected matrix: \n{selected_matrix}")
            solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

        # Forward Gauss-Seidel Method
        elif flag == 2:
            print(f"\nUsing Forward Gauss-Seidel Method with selected matrix: \n{selected_matrix}")
            solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

        # Backward Gauss-Seidel Method
        elif flag == 3:
            print(f"\nUsing Backward Gauss-Seidel Method with selected matrix: \n{selected_matrix}")
            solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

        # Symmetric Gauss-Seidel Method
        else: # flag == 4
            print(f"\nUsing Symmetric Gauss-Seidel Method with selected matrix: \n{selected_matrix}")
            solution, iteration, relative_error = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, flag)

        # Appending lists that are the output of the methods
        solution_list.append(solution)
        iteration_list.append(iteration)
        relative_error_list.append(relative_error)

        # Computing the matrix G from ||Ge_k|| for the method selected and appending the list for each initial guess
        G_matrix = G_error_matrix(selected_matrix, k, flag)
        G_matrix_list.append(G_matrix)

        # Computing the 2-norm of the G matrix and appending its list for each initial guess
        G_matrix_norm = np.linalg.norm(G_matrix, 2)
        G_matrix_norms_list.append(G_matrix_norm)

        # Computes the spectral radius of G and appending the list for each initial guess
        spectral_radius = spectral_radi(G_matrix)
        spectral_radius_list.append(spectral_radius)

        # Debug Statements for each run of the for loop (each solution vector)
        if debug:
            print(f"Results for Solution Vector #{i}:")
            print(f"Solution: {solution}")
            print(f"Iterations for convergence: {iteration}")
            print(f"Relative Error: {relative_error}")
            print(f"G Error Matrix: \n{G_matrix}")
            print(f"Spectral Radius of G is: {spectral_radius}")

    # Printing Overall results
    print(f"\nOverall Results for {g} solution vectors:")
    #print(f"Solution vectors are: {solution_list}")
    print(f"Number of iterations for each solution vector: {iteration_list}")
    #print(f"Relative errors are: {relative_error_list}")
    #print(f"G error matrices are: {G_matrix_list}")
    print(f"Spectral Radii of G's are: {spectral_radius_list}")
    print(f"2-Norm of the G matrices are: {G_matrix_norms_list}")

    return solution_list, iteration_list, relative_error_list

# Driver function for the methods
def part_2_driver_multiple(selected_matrix, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug):
    # Getting user inputs
    #inputs = get_user_inputs()

    #selected_matrix, smin, smax, g, tol, max_iter, debug, ini, flag = inputs

    # Sets random seed for reproducibility
    #np.random.seed(42)

    # Setting Initial Conditions
    rows, cols = selected_matrix.shape
    k = rows

    if debug:
        print(f"Matrix Selected is: \n{selected_matrix}")
        print(f"x0 is: {x0}")

    # Initializing lists to hold values for output
    # Lists for Jacobi
    solution_list_jac = []
    iteration_list_jac = []
    relative_error_list_jac = []
    spectral_radius_list_jac = []
    G_matrix_norms_list_jac = []

    # Lists for FGS
    solution_list_fgs = []
    iteration_list_fgs = []
    relative_error_list_fgs = []
    spectral_radius_list_fgs = []
    G_matrix_norms_list_fgs = []

    # Lists for BGS
    solution_list_bgs = []
    iteration_list_bgs = []
    relative_error_list_bgs = []
    spectral_radius_list_bgs = []
    G_matrix_norms_list_bgs = []

    # Lists for SGS
    solution_list_sgs = []
    iteration_list_sgs = []
    relative_error_list_sgs = []
    spectral_radius_list_sgs = []
    G_matrix_norms_list_sgs = []

    print(f"\nUsing each method with the selected matrix: \n{selected_matrix}")
    # For loop running over multiple solution vectors for same matrix and initial guess vector
    for i in range(g):
        # Setting a new x_tilde (solution vector) time to test over multiple solution vectors for same matrix and same initial guess vector
        x_tilde = generate_float_1D_vector_np(smin, smax, k)
        b = np.dot(selected_matrix, x_tilde)

        for j in range(num_x0):
            # Generate Random Initial Guess Vectors
            x0 = generate_float_1D_vector_np(dmin, dmax, k)

            # Debug print statements for initial guess, true solution, and b vectors
            if debug:
                print(f"Initial Guess Vector x0 = {x0}")
                print(f"True Solution Vector x_tilde = {x_tilde}")
                print(f"Ax = b is: {b}")

            # Jacobi Method
            solution_jac, iteration_jac, relative_error_jac = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, 1)

            # Forward Gauss-Seidel Method
            solution_fgs, iteration_fgs, relative_error_fgs = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, 2)

            # Backward Gauss-Seidel Method
            solution_bgs, iteration_bgs, relative_error_bgs = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, 3)

            # Symmetric Gauss-Seidel Method
            solution_sgs, iteration_sgs, relative_error_sgs = stationary_method(selected_matrix, b, x0, x_tilde, tol, max_iter, 4)

            # Appending lists that are the output of the methods
            # Lists for outputs for Jacobi
            solution_list_jac.append(solution_jac)
            iteration_list_jac.append(iteration_jac)
            relative_error_list_jac.append(relative_error_jac)

            # Lists for outputs for FGS
            solution_list_fgs.append(solution_fgs)
            iteration_list_fgs.append(iteration_fgs)
            relative_error_list_fgs.append(relative_error_fgs)

            # Lists for outputs for BGS
            solution_list_bgs.append(solution_bgs)
            iteration_list_bgs.append(iteration_bgs)
            relative_error_list_bgs.append(relative_error_bgs)

            # Lists for outputs for SGS
            solution_list_sgs.append(solution_sgs)
            iteration_list_sgs.append(iteration_sgs)
            relative_error_list_sgs.append(relative_error_sgs)

        # Computing the matrix G from ||Ge_k|| for the method selected and appending the list for each initial guess
        G_matrix_jac = G_error_matrix(selected_matrix, k, 1)
        G_matrix_fsg = G_error_matrix(selected_matrix, k, 2)
        G_matrix_bgs = G_error_matrix(selected_matrix, k, 3)
        G_matrix_sgs = G_error_matrix(selected_matrix, k, 4)

        # Computing the 2-norm of the G matrix and appending its list for each initial guess
        G_matrix_norm_jac = np.linalg.norm(G_matrix_jac, 2)
        G_matrix_norm_fsg = np.linalg.norm(G_matrix_fsg, 2)
        G_matrix_norm_bgs = np.linalg.norm(G_matrix_bgs, 2)
        G_matrix_norm_sgs = np.linalg.norm(G_matrix_sgs, 2)
        G_matrix_norms_list_jac.append(G_matrix_norm_jac)
        G_matrix_norms_list_fgs.append(G_matrix_norm_fsg)
        G_matrix_norms_list_bgs.append(G_matrix_norm_bgs)
        G_matrix_norms_list_sgs.append(G_matrix_norm_sgs)

        # Computes the spectral radius of G and appending the list for each initial guess
        spectral_radius_jac = spectral_radi(G_matrix_jac)
        spectral_radius_fsg = spectral_radi(G_matrix_fsg)
        spectral_radius_bgs = spectral_radi(G_matrix_bgs)
        spectral_radius_sgs = spectral_radi(G_matrix_sgs)
        spectral_radius_list_jac.append(spectral_radius_jac)
        spectral_radius_list_fgs.append(spectral_radius_fsg)
        spectral_radius_list_bgs.append(spectral_radius_bgs)
        spectral_radius_list_sgs.append(spectral_radius_sgs)

        # Debug Statements for each run of the for loop (each solution vector)
        # if debug:
        #     print(f"Results for Solution Vector #{i}:")
        #     print(f"Solution: {solution}")
        #     print(f"Iterations for convergence: {iteration}")
        #     print(f"Relative Error: {relative_error}")
        #     print(f"G Error Matrix: \n{G_matrix}")
        #     print(f"Spectral Radius of G is: {spectral_radius}")

    reshape_iteration_list_result = {
        "Jacobi": np.array(iteration_list_jac).reshape(g, num_x0),
        "FGS": np.array(iteration_list_fgs).reshape(g, num_x0),
        "BGS": np.array(iteration_list_bgs).reshape(g, num_x0),
        "SGS": np.array(iteration_list_sgs).reshape(g, num_x0),
    }


    print(f"Iterations Jacobi: {iteration_list_jac}")
    print(f"Iterations FGS: {iteration_list_fgs}")
    print(f"Iterations BGS: {iteration_list_bgs}")
    print(f"Iterations SGS: {iteration_list_sgs}")
    print(f"G Matrix Norms Jacobi: {G_matrix_norms_list_jac}")
    print(f"G Matrix Norms FGS: {G_matrix_norms_list_fgs}")
    print(f"G Matrix Norms BGS: {G_matrix_norms_list_bgs}")
    print(f"G Matrix Norms SGS: {G_matrix_norms_list_sgs}")
    print(f"Spectral Radii Jacobi: {spectral_radius_list_jac}")
    print(f"Spectral Radii FGS: {spectral_radius_list_fgs}")
    print(f"Spectral Radii BGS: {spectral_radius_list_bgs}")
    print(f"Spectral Radii SGS: {spectral_radius_list_sgs}")

    # Sets x-axis labels for number of initial guess vectors
    x_labels = [f"x0_{i}" for i in range(1, num_x0 + 1)]

    # Randomly Choosing 3 Solution Vectors from however many were tested defined by the user, "g"
    random_indices = np.random.choice(g, 3, replace=False)

    # Creating the 3 subplots that are horizontal instead of stacked
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    # Checks the case if the axes is not iterable
    if g == 1:
        axes = [axes]

    # Loops through each randomly selected solution vector from above
    for i, ax in enumerate(axes):
        # Gets the index from random_indices for the solution vector
        index = random_indices[i]

        # Plot Each Method's Result on the same graph for the given solution vector
        for method, iteration_lists in reshape_iteration_list_result.items():
            ax.plot(x_labels, iteration_lists[index], label=method)

        # Changing the x-axis ticks to be every 3
        ax.set_xticks(x_labels[::3])

        # Changes the x-axis labels to be every 3 and rotates them for readability
        ax.set_xticklabels(x_labels[::3], rotation=45)

        # Sets the labels, grid, and titles of the subplots
        ax.set_xlabel("Initial Guess Vectors")
        ax.set_ylabel("Iterations")
        ax.set_title(f"Iterations for x_tilde {index+1}")
        ax.grid(True)
        ax.legend(loc="best")

    # Shows the plot and gives the overall title of the plot
    plt.suptitle("Convergence for Each Method with Given Solution Vectors and Initial Guess Vectors")
    plt.tight_layout()
    plt.show()


    # # Create Subplots
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    #
    # # 1. Plot for Iterations to Converge (Top-left)
    # ax1.plot(iteration_list_jac, label="Jacobi", marker='o')
    # ax1.plot(iteration_list_fgs, label="Forward Gauss-Seidel", marker='x')
    # ax1.plot(iteration_list_bgs, label="Backward Gauss-Seidel", marker='^')
    # ax1.plot(iteration_list_sgs, label="Symmetric Gauss-Seidel", marker='s')
    # ax1.set_xlabel("Test Run")
    # ax1.set_ylabel("Iterations to Converge")
    # ax1.set_title("Iterations to Converge for Each Method")
    # ax1.legend()
    # ax1.grid(True)
    #
    # # # 2. Plot for Relative Error (Top-right)
    # # axes[0, 1].plot(relative_error_list_jac, label="Jacobi", marker='o')
    # # axes[0, 1].plot(relative_error_list_fgs, label="Forward Gauss-Seidel", marker='x')
    # # axes[0, 1].plot(relative_error_list_bgs, label="Backward Gauss-Seidel", marker='^')
    # # axes[0, 1].plot(relative_error_list_sgs, label="Symmetric Gauss-Seidel", marker='s')
    # # axes[0, 1].set_xlabel("Test Run")
    # # axes[0, 1].set_ylabel("Relative Error")
    # # axes[0, 1].set_title("Relative Error for Each Method")
    # # axes[0, 1].legend()
    # # axes[0, 1].grid(True)
    #
    # # 3. Plot for G Matrix Norms (Bottom-left)
    # ax2.plot(G_matrix_norms_list_jac, label="Jacobi", marker='o')
    # ax2.plot(G_matrix_norms_list_fgs, label="Forward Gauss-Seidel", marker='x')
    # ax2.plot(G_matrix_norms_list_bgs, label="Backward Gauss-Seidel", marker='^')
    # ax2.plot(G_matrix_norms_list_sgs, label="Symmetric Gauss-Seidel", marker='s')
    # ax2.set_xlabel("Test Run")
    # ax2.set_ylabel("2-Norm of G Matrix")
    # ax2.set_title("2-Norm of G Matrix for Each Method")
    # ax2.legend()
    # ax2.grid(True)
    #
    # # 4. Plot for Spectral Radius of G (Bottom-right)
    # ax3.plot(spectral_radius_list_jac, label="Jacobi", marker='o')
    # ax3.plot(spectral_radius_list_fgs, label="Forward Gauss-Seidel", marker='x')
    # ax3.plot(spectral_radius_list_bgs, label="Backward Gauss-Seidel", marker='^')
    # ax3.plot(spectral_radius_list_sgs, label="Symmetric Gauss-Seidel", marker='s')
    # ax3.set_xlabel("Test Run")
    # ax3.set_ylabel("Spectral Radius of G")
    # ax3.set_title("Spectral Radius of G for Each Method")
    # ax3.legend()
    # ax3.grid(True)
    #
    # # Adjust layout to avoid overlap
    # plt.suptitle(f"Plots for {selected_matrix_key} with {g} solution vectors ", fontsize=16)
    # plt.tight_layout()
    # plt.show()


    return (solution_list_jac, solution_list_fgs, solution_list_bgs, solution_list_sgs,
            iteration_list_jac, iteration_list_fgs, iteration_list_bgs, iteration_list_sgs)

# Main function
if __name__ == "__main__":
    while True:
        inputs = get_user_inputs()
        selected_matrix, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag = inputs

        if not ini:
            (solution_list_jac, solution_list_fgs, solution_list_bgs, solution_list_sgs,
             iteration_list_jac, iteration_list_fgs,
             iteration_list_bgs, iteration_list_sgs) = part_2_driver_multiple(selected_matrix,
                                                                           smin, smax, g, dmin, dmax, num_x0, tol,
                                                                           max_iter, debug)
        else:
            solution, iteration, relative_error_list = part_2_driver_one(selected_matrix, smin, smax,
                                                                     g, tol, max_iter, debug, flag)

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("Thank you for using the Solver!")
