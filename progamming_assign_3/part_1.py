import numpy as np
import matplotlib.pyplot as plt
from my_package import generate_float_1D_vector_np, vec_2_norm_np, generate_float_1D_uniform_vector_np, \
    generate_float_normal_vector_np


## RF
def richard_first(A, b, x0, x_tilde, tol, max_iter):
    """
        Richardson Iteration Method Function:
        Parameters:
            A: Matrix or Vector to be examined
            b: Solution of Ax = b
            x0: Prediction Vector of Solution
            x_tilde: True Solution of Ax = b
            tol: Error tolerance
            max_iter: Maximum number of iterations

        Returns:
            x: Solution vector
            iter_num: Number of iterations needed for convergence below error tolerance
            residual_list: List of residuals at each iteration
            err_list: List of errors at each iteration
    """

    # Setting alpha_opt
    alpha = 2 / (np.min(A) + np.max(A))

    # Setting x to the prediction vector
    x = x0

    # Setting the true solution
    x_true = x_tilde

    # Calculating Initial Error 2-Norm
    err = vec_2_norm_np(x - x_true)
    err_list = []

    # Setting initial residual and initial residual norm
    r = b - A * x
    r0_norm = vec_2_norm_np(r)
    residual_list = [r0_norm]

    # Initial Iteration
    iter_num = 0

    # Performing Richardson's Iteration
    while iter_num < max_iter:
        # Update the solution
        x_next = x + alpha * r

        err_next = vec_2_norm_np(x_next - x_true)
        rel_err = err_next / err
        err_list.append(rel_err)

        # Updating the residual
        r_next = r - alpha * A * r

        # Check for Convergence by seeing if relative error < error tolerance
        r_next_norm = vec_2_norm_np(r_next)
        residual_list.append(r_next_norm)
        if r_next_norm / r0_norm < tol:
            return x_next, iter_num + 1, residual_list, err_list

        # Setting up for next iteration if convergence has not hit
        x = x_next
        err = err_next
        r = r_next
        iter_num += 1

    return x, iter_num, residual_list, err_list

## SD
def steep_descent(A, b, x0, x_tilde, tol, max_iter):
    """
        Steepest Descent Method Function:
        Parameters:
            A: Matrix or Vector to be examined
            b: Solution of Ax = b
            x0: Prediction Vector of Solution
            x_tilde: True Solution of Ax = b
            tol: Error tolerance
            max_iter: Maximum number of iterations

        Returns:
            x: Solution vector
            iter_num: Number of iterations needed for convergence below error tolerance
            residual_list: List of residuals at each iteration
            err_list: List of errors at each iteration
    """

    # Setting initial values for solution, residual, v, and iteration
    x = x0
    x_true = x_tilde
    err0 = x - x_true
    err01 = A * err0
    err = np.dot(err0, err01)
    err_list = []

    r = b - A * x
    r0_norm = vec_2_norm_np(r)
    residual_list = [r0_norm]
    v = A * r
    iter_num = 0

    # Performing Steepest Descent Method
    while iter_num < max_iter:
        # Setting alpha each iteration
        alpha = (np.dot(r, r))/(np.dot(r, v))

        # Updating the next solution
        x_next = x + alpha * r

        # Updating the Error and Appending to the list to store at each iteration
        err_next0 = x_next - x_true
        err_next1 = A * err_next0
        err_next_full = np.dot(err_next0, err_next1)
        rel_err = err_next_full / err
        err_list.append(rel_err)

        # Updating the residual
        r_next = r - v * alpha

        # Checking for convergence by computing relative error < error tolerance set
        r_next_norm = vec_2_norm_np(r_next)
        residual_list.append(r_next_norm)
        if r_next_norm / r0_norm < tol:
            return x_next, iter_num + 1, residual_list, err_list

        # Updating v
        v_next = A * r_next

        # Preparing next iteration values
        x = x_next
        err = err_next_full
        r = r_next
        v = v_next
        iter_num += 1

    return x, iter_num, residual_list, err_list

## CG
def conj_grad(A, b, x0, x_tilde, tol, max_iter):
    """
    Conjugate Gradient Method Function:
    Parameters:
        x0: Prediction Vector of Solution
        A: Matrix or Vector to be examined
        b: Solution of Ax = b
        x_tilde: True Solution of Ax = b
        tol: Error tolerance
        max_iter: Maximum number of iterations

    Returns:
        x: Solution vector
        iter_num: Number of iterations needed for convergence below error tolerance
    """

    x = x0
    x_true = x_tilde
    r = b - (A * x)
    r0_norm = vec_2_norm_np(r)
    residual_list = [r0_norm]
    d = r
    sigma = np.dot(r, r)
    err0 = x - x_true
    err01 = A * err0
    err = np.dot(err0, err01)
    err_list = []
    iter_num = 0

    while iter_num < max_iter:
        v = A * d
        mu = np.dot(d, v)
        alpha = sigma/mu
        x_next = x + alpha * d
        r_next = r - alpha * v

        # Error for Conjugate Gradient
        err_next0 = x_next - x_true
        err_next1 = A * err_next0
        err_next_full = np.dot(err_next0, err_next1)
        rel_err = err_next_full / err
        err_list.append(rel_err)

        r_next_norm = vec_2_norm_np(r_next)
        residual_list.append(r_next_norm)
        if r_next_norm / r0_norm < tol:
            return x_next, iter_num + 1, residual_list, err_list

        sigma_next = np.dot(r_next, r_next)
        beta = sigma_next/sigma
        d_next = r_next + beta * d

        x = x_next
        r = r_next
        sigma = sigma_next
        d = d_next
        err = err_next_full
        iter_num += 1

    return x, iter_num, residual_list, err_list

# Distinct Eigenvalues with Random Multiplicities function
def create_distinct_eigenvalues(n, k, lambda_min, lambda_max):
    """Create k distinct eigenvalues with random multiplicities"""
    # Generate k distinct eigenvalues from lambda_min to lambda_max across uniform distribution
    eigenvals = np.random.uniform(lambda_min, lambda_max, k)

    # Generate multiplicities of the eigenvalues to fill in rest, n - k
    multiplicities = np.random.multinomial(n, [1/k] * k)

    # Create the diagonal vector using the eigenvalues and random multiplicities
    diag = np.repeat(eigenvals, multiplicities)

    return diag, eigenvals, multiplicities

# Distinct Clusters of Eigenvalues function
def create_distinct_cluster(n, k, lambda_min, lambda_max, var_type):
    # Generate k distinct eigenvalues from lambda_min to lambda_max across uniform distribution
    eigenvals = np.random.uniform(lambda_min, lambda_max, k)

    # Generate multiplicities of the eigenvalues to fill in rest, n - k
    multiplicities = np.random.multinomial(n, [1 / k] * k)

    # Create the diagonal list that is created from k distinct clusters and values taken from normal distribution
    diagonal = []
    for i in range(k):
        if var_type == 1:
            clust_var = 0.01 * (lambda_max - lambda_min)
            # Generates the clusters of values for each k distinct eigenvalue and given multiplicity
            cluster = np.random.normal(eigenvals[i], clust_var, multiplicities[i])

            # Add all elements of cluster to diagonal at once (avoids for loop for appending)
            diagonal.extend(cluster)

        elif var_type == 2:
            clust_var = 0.1 * (lambda_max - lambda_min)
            # Generates the clusters of values for each k distinct eigenvalue and given multiplicity
            cluster = np.random.normal(eigenvals[i], clust_var, multiplicities[i])

            # Add all elements of cluster to diagonal at once (avoids for loop for appending)
            diagonal.extend(cluster)

        else:
            clust_var = 0.25 * (lambda_max - lambda_min)
            # Generates the clusters of values for each k distinct eigenvalue and given multiplicity
            cluster = np.random.normal(eigenvals[i], clust_var, multiplicities[i])

            # Add all elements of cluster to diagonal at once (avoids for loop for appending)
            diagonal.extend(cluster)

    diagonal = np.array(diagonal)

    return diagonal, eigenvals, multiplicities

## User Input Function
def get_user_inputs():
    """Get problem parameters from user input.

    Prompts user for:
    - Matrix dimensions (n rows x n columns)
    - Problem type (simple test, uniform random, normal random)
    - Range for random values (dmin, dmax)
    - Debug output preference
    """

    print("\nRF, SD, CD Construction")
    print("----------------------------------")

    while True:
        try:
            n = int(input("Enter number of rows (n) [default=10]: ") or "10")

            print("\nChoose problem type:")
            print("1. All Eigenvalues the same")
            print("2. k distinct eigenvalues with multiplicities")
            print("3. k distinct eigenvalues with random distributions around each")
            print("4. Eigenvalues chosen from a Uniform Distribution, specified lambda_min and lambda_max")
            print("5. Eigenvalues chosen from a Normal Distribution, specified lambda_min and lambda_max")
            problem_type = int(input("Enter problem type (1-5) [default=1]: ") or "1")

            if problem_type not in [1, 2, 3, 4, 5]:
                print("Error: Invalid problem type")
                continue

            # New inputs for random reflector range
            print("\nSet range for random values of Solution Vectors and Initial Guess Vectors:")
            dmin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
            dmax = float(input("Enter maximum value (default=10.0): ") or "10.0")

            if dmin >= dmax:
                print("Error: Minimum value must be less than maximum value")
                continue

            print("\nChoose How Many Initial Guess Vectors:")
            g = int(input("Enter number of Initial Guess Vectors (default=10): ") or "10")

            print("\nSet Tolerance Level for Convergence to Hit:")
            tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")

            print("\nSet Maximum Number of Iterations to be Ran:")
            max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")

            lambda_min, lambda_max, distinct_eig, var_type = None, None, None, None

            if problem_type in [2, 3]:
                print("\nChoose how many distinct eigenvalues to use:")
                distinct_eig = int(input("Enter number of distinct eigenvalues (default=5): ") or "5")

            if problem_type in [2, 3, 4, 5]:
                print("\nChoose Minimum and Maximum for Eigenvalues (Must be positive): ")
                lambda_min = float(input("Enter lambda min (default=1.0): ") or "1.0")
                lambda_max = float(input("Enter lambda max (default=10.0): ") or "10.0")
                if lambda_max <= lambda_min:
                    print("Error: Maximum value must be less than minimum value")
                    continue

            if problem_type in [3]:
                print("\nChoose variance of clustering from Normal Distribution (Relative to Eigenvalue Range):")
                print("1. Tight Clustering/Spread")
                print("2. Moderate Clustering/Spread")
                print("3. Large Clustering/Spread")
                var_type = int(input("Enter cluster variance (1-3) (default=1): ") or "1")

            debug = input("\nEnable debug output? (y/n) [default=n]: ").lower().startswith('y')

            return n, problem_type, dmin, dmax, g, tol, max_iter, distinct_eig, lambda_min, lambda_max, var_type, debug

        except ValueError:
            print("Error: Please enter valid numbers")

def part_1_driver():
    # Setting User Inputs
    inputs = get_user_inputs()

    n, problem_type, dmin, dmax, g, tol, max_iter, distinct_eig, lambda_min, lambda_max, var_type, debug = inputs

    # Sets seed for reproducibility
    #np.random.seed(42)

    # All eigenvalues the same
    if problem_type == 1:
        Lambda = generate_float_1D_vector_np(5, 5, n)     # Eigenvalue, diagonal matrix (created as a vector)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde  # Lambda * x_tilde

    # k distinct eigenvalues with random multiplicities
    elif problem_type == 2:
        Lambda, eigvals, multiplic = create_distinct_eigenvalues(n, distinct_eig, lambda_min, lambda_max)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde
        if debug:
            print("\nProblem type 2 values:")
            print(f"Diagonal Vector Lambda = {Lambda}")
            print(f"Distinct Eigenvalues: {eigvals}")
            print(f"Multiplicities of Eigenvalues: {multiplic}")

    # k distinct eigenvalues with random distributions around each k distinct eigenvalue
    elif problem_type == 3:
        Lambda, eigvals, multiplic = create_distinct_cluster(n, distinct_eig, lambda_min, lambda_max, var_type)   # Eigenvalue, diagonal matrix (created as a vector)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde
        if debug:
            print("\nProblem type 3 values:")
            print(f"Diagonal Vector Lambda = {Lambda}")
            print(f"Distinct Eigenvalues: {eigvals}")
            print(f"Multiplicities of Eigenvalues: {multiplic}")

    # Eigenvalues generated from a Uniform Distribution (specified lambda_min, lambda_max) for conditioning purposes
    elif problem_type == 4:
        Lambda = generate_float_1D_uniform_vector_np(lambda_min, lambda_max, n)     # Eigenvalue, diagonal matrix (created as a vector)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde

    # Eigenvalues generated from a Normal Distribution
    else: # problem_type == 5
        Lambda = generate_float_normal_vector_np(lambda_min, lambda_max, n)     # Eigenvalue, diagonal matrix (created as a vector)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde


    # Printing Lambda, x_tilde, and b_tilde
    if debug:
        print(f"\nMatrix Lambda: {Lambda}")
        print(f"\nMatrix x_tilde is: {x_tilde}")
        print(f"\nMatrix b_tilde is: {b_tilde}")

    # Bounds and Condition Number for Richardson's, SD, and CG
    kappa = np.max(Lambda)/np.min(Lambda)
    bound_rich_steep = (kappa - 1)/(kappa + 1)
    bound_conj = (np.sqrt(kappa) - 1)/(np.sqrt(kappa) + 1)

    # Initializing lists to append the solutions and iterations from output below
    solution_richard = []
    solution_steep = []
    solution_conj = []
    iteration_richard = []
    iteration_steep = []
    iteration_conj = []

    # Creating Subplots for Errors that will be added each iteration and each initial guess
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

    # Richardson Error Plot Initialization
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Relative Error")
    ax1.set_title("Relative Error for Richardson")
    ax1.axhline(y = bound_rich_steep, linestyle = "--", color = "black")
    ax1.grid(True)

    # Steepest Descent Error Plot Initialization
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Relative Error for Steepest Descent")
    ax2.axhline(y = bound_rich_steep, linestyle = "--", color = "black")
    ax2.grid(True)

    # Conjugate Gradient Error Plot Initialization
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Relative Error")
    ax3.set_title("Relative Error for Conjugate Gradient")
    ax3.axhline(y = bound_conj, linestyle = "--", color = "black")
    ax3.grid(True)

    fig.suptitle(f"Relative Errors for Each Method under Problem Type: {problem_type}")

    # g-number of initial guesses to pass through each method to compare for each x_tilde, b_tilde, Lambda
    for i in range(g):
        # Initial Guess Vector
        x0 = np.random.uniform(dmin,dmax,n)

        # Computing Each method for each Initial Guess Vector
        solution_1, iterations_1, residuals_richardson, errors_richardson = richard_first(Lambda, b_tilde, x0, x_tilde,
                                                                                          tol, max_iter)
        solution_2, iterations_2, residuals_steep_descent, errors_steep_descent = steep_descent(Lambda, b_tilde, x0,
                                                                                                x_tilde, tol, max_iter)
        solution_3, iterations_3, residuals_conj_grad, errors_conj_grad = conj_grad(Lambda, b_tilde, x0, x_tilde, tol,
                                                                                    max_iter)

        # Creating x-axis, number of iterations, for plots below
        steps_rich = range(len(errors_richardson))
        steps_steep = range(len(errors_steep_descent))
        steps_conj = range(len(errors_conj_grad))

        # Richardson Error plot with log values of errors taken
        ax1.semilogy(steps_rich, errors_richardson)#, label="Richardson's Method" if i == 0 else "")

        # Steepest Descent error plot with log values of errors taken
        ax2.semilogy(steps_steep, errors_steep_descent)#, label="Steepest Descent" if i == 0 else "")

        # Conjugate Gradient error plot with log values of errors taken
        ax3.semilogy(steps_conj, errors_conj_grad)#, label="Conjugate Gradient" if i == 0 else "")

        # Appending lists from the outputs of the methods for solution vector and number of iterations taken
        solution_richard.append(solution_1)
        solution_steep.append(solution_2)
        solution_conj.append(solution_3)
        iteration_richard.append(iterations_1)
        iteration_steep.append(iterations_2)
        iteration_conj.append(iterations_3)

        # Debug Output
        if debug:
            print(f"\nInitial Guess Vector Iteration: {i+1}:")
            print(f"Initial Guess Vector: {x0}")
            print(f"Richardson Solution Vector: {solution_1}")
            print(f"Richardson Error List: {errors_richardson}")
            print(f"Steepest Descent Solution Vector: {solution_2}")
            print(f"Steepest Descent Error List: {errors_steep_descent}")
            print(f"Conjugate Gradient Guess Vector: {solution_3}")
            print(f"Conjugate Gradient Error List: {errors_conj_grad}")


    # Showing the plot
    plt.show()

    # Creating plots for amount of iterations taken for each initial guess vector to converge for each method #
    # Ticks is created to have integers on x-axis
    ticks = range(g+1)

    # Creating the figure for each method plotted on the same graph with x-axis as initial guess vector and y-axis as
    # number of iterations taken to converge for given initial guess vector
    plt.figure(figsize=(10,6))
    plt.plot(range(len(iteration_richard)), iteration_richard, color = 'blue', marker = "o", label="Richardson's Method")
    plt.plot(range(len(iteration_steep)), iteration_steep, color = 'red', marker = "x", label="Steepest Descent")
    plt.plot(range(len(iteration_conj)), iteration_conj, color = 'green', marker = "s", label="Conjugate Gradient")
    plt.xticks(ticks)
    plt.xlabel("Initial Guess Vector")
    plt.ylabel("Number of Iterations")
    plt.title(f"Number of Iterations to Hit Convergence for Richardson's, SD, and CG with Problem Type {problem_type} Selected")
    plt.legend(title="Method", loc = "best")
    plt.grid(True)
    plt.show()

    # # Print Results for solution, number of iterations, and error of converged solution and true solution
    # if debug:
    #     print("\nResults:")
    #     print("---------")
    #     solutions = {
    #         'Richardson Iteration Solution' : solution_1,
    #         'Steepest Descent Iteration Solution': solution_2,
    #         'Conjugate Gradient Iteration Solution': solution_3,
    #     }
    #     iterations = {
    #         'Number of Iterations for Richardson`s Method' : iterations_1,
    #         'Number of Iterations for Steepest Descent': iterations_2,
    #         'Number of Iterations for Conjugate Gradient': iterations_3,
    #     }
    #     errors = {
    #         'Final Richardson error': vec_2_norm_np(solution_1 - x_tilde),
    #         'Final Steepest Descent Error': vec_2_norm_np(solution_2 - x_tilde),
    #         'Final Conjugate Gradient Error': vec_2_norm_np(solution_3 - x_tilde)
    #     }
    #     for name, solution in solutions.items():
    #         print(f'{name}: {solution}')
    #     for name, iterations in iterations.items():
    #         print(f'{name}: {iterations}')
    #     for name, error in errors.items():
    #         print(f'{name}: {error}')


    # Setting Bounds and Condition Number


    return solution_richard, solution_steep, solution_conj, iteration_richard, iteration_steep, iteration_conj

# Main function that will have the outputs from the methods and asks user if another run is wanting to be taken
if __name__ == "__main__":
    while True:
        solution_richard, solution_steep, solution_conj, iteration_richard, iteration_steep, iteration_conj = part_1_driver()

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("Thank you for using the Iteration Solver!")


