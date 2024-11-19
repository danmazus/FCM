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
    err = vec_2_norm_np(x - x_true)
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
        err_next = vec_2_norm_np(x_next - x_true)
        rel_err = err_next / err
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
        err = err_next
        r = r_next
        v = v_next
        iter_num += 1

    return x, iter_num, residual_list, err_list

## CG
def conj_grad(A, b, x0, tol, max_iter):
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
    r = b - (A * x)
    r0_norm = vec_2_norm_np(r)
    residual_list = [r0_norm]
    d = r
    sigma = np.dot(r, r)
    iter_num = 0

    while iter_num < max_iter:
        v = A * d
        mu = np.dot(d, v)
        alpha = sigma/mu
        x_next = x + alpha * d
        r_next = r - alpha * v

        r_next_norm = vec_2_norm_np(r_next)
        residual_list.append(r_next_norm)
        if r_next_norm / r0_norm < tol:
            return x_next, iter_num + 1, residual_list

        sigma_next = np.dot(r_next, r_next)
        beta = sigma_next/sigma
        d_next = r_next + beta * d

        x = x_next
        r = r_next
        sigma = sigma_next
        d = d_next
        iter_num += 1

    return x, iter_num, residual_list

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

            print("\nSet Tolerance Level for Convergence to Hit:")
            tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")

            print("\nSet Maximum Number of Iterations to be Ran:")
            max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")

            lambda_min, lambda_max = None, None

            if problem_type in [4, 5]:
                print("\nChoose Minimum and Maximum for Eigenvalues (Must be positive): ")
                lambda_min = float(input("Enter lambda min (default=1.0): ") or "1.0")
                lambda_max = float(input("Enter lambda max (default=10.0): ") or "10.0")
                if lambda_max <= lambda_min:
                    print("Error: Maximum value must be less than minimum value")
                    continue

            debug = input("\nEnable debug output? (y/n) [default=n]: ").lower().startswith('y')

            return n, problem_type, dmin, dmax, tol, max_iter, lambda_min, lambda_max, debug

        except ValueError:
            print("Error: Please enter valid numbers")

def part_1_driver():
    # Setting User Inputs
    inputs = get_user_inputs()

    n, problem_type, dmin, dmax, tol, max_iter, lambda_min, lambda_max, debug = inputs

    # Sets seed for reproducibility
    #np.random.seed(42)

    # All eigenvalues the same
    if problem_type == 1:
        Lambda = generate_float_1D_vector_np(5, 5, n)     # Eigenvalue, diagonal matrix (created as a vector)
        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde  # Lambda * x_tilde

    # k distinct eigenvalues with random multiplicities
    elif problem_type == 2:
        Lambda = np.zeros(n)     # Eigenvalue, diagonal matrix (created as a vector)
        #for i in range(n):

        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde

    # k distinct eigenvalues with random distributions around each k distinct eigenvalue
    elif problem_type == 3:
        Lambda = np.zeros(n)    # Eigenvalue, diagonal matrix (created as a vector)
        #for i in range(n):

        x_tilde = generate_float_1D_vector_np(dmin, dmax, n)    # Random Solution Vector
        b_tilde = Lambda * x_tilde   # Lambda * x_tilde

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

    # Initial Guess Vector x0
    x0 = np.random.randn(n)

    if debug:
        print(f"\nInitial Guess x0 is: {x0}")


    # Computing each method with resulting solution, # of iterations, and list of residuals for plotting at each iteration
    solution_1, iterations_1, residuals_richardson, errors_richardson = richard_first(Lambda, b_tilde, x0, x_tilde, tol, max_iter)
    solution_2, iterations_2, residuals_steep_descent, errors_steep_descent = steep_descent(Lambda, b_tilde, x0, x_tilde, tol, max_iter)
    solution_3, iterations_3, residuals_conj_grad = conj_grad(Lambda, b_tilde, x0, tol, max_iter)

    # Print Results for solution, number of iterations, and error of converged solution and true solution
    if debug:
        print("\nResults:")
        print("---------")
        solutions = {
            'Richardson Iteration Solution' : solution_1,
            'Steepest Descent Iteration Solution': solution_2,
            'Conjugate Gradient Iteration Solution': solution_3,
        }
        iterations = {
            'Number of Iterations for Richardson`s Method' : iterations_1,
            'Number of Iterations for Steepest Descent': iterations_2,
            'Number of Iterations for Conjugate Gradient': iterations_3,
        }
        errors = {
            'Final Richardson error': vec_2_norm_np(solution_1 - x_tilde),
            'Final Steepest Descent Error': vec_2_norm_np(solution_2 - x_tilde),
            'Final Conjugate Gradient Error': vec_2_norm_np(solution_3 - x_tilde)
        }
        for name, solution in solutions.items():
            print(f'{name}: {solution}')
        for name, iterations in iterations.items():
            print(f'{name}: {iterations}')
        for name, error in errors.items():
            print(f'{name}: {error}')


    # Setting Bounds and Condition Number
    kappa = np.max(Lambda)/np.min(Lambda)
    bound_rich_steep = (kappa - 1)/(kappa + 1)
    bound_conj = (np.sqrt(kappa) - 1)/(np.sqrt(kappa) + 1)


    """ PLOTS """
    # Plot the residuals of each method
    plt.figure(figsize=(24,6))

    # Richardson's Method
    plt.subplot(1, 3, 1)
    plt.semilogy(residuals_richardson, label="Residual Norm")
    #plt.plot(residuals_1, label="Residual Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    #plt.ylabel("Residual Norm")
    plt.title("Convergence of Richardson Iteration")
    plt.legend()
    plt.grid(True)

    # Steepest Descent Method
    plt.subplot(1, 3, 2)
    plt.semilogy(residuals_steep_descent, label="Residual Norm")
    #plt.plot(residuals_2, label="Residual Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    #plt.ylabel("Residual Norm")
    plt.title("Convergence of Steepest Descent Iteration")
    plt.legend()
    plt.grid(True)

    # Conjugate Gradient Method
    plt.subplot(1, 3, 3)
    plt.semilogy(residuals_conj_grad, label="Residual Norm")
    #plt.plot(residuals_3, label="Residual Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm (log scale)")
    #plt.ylabel("Residual Norm")
    plt.title("Convergence of Conjugate Gradient Iteration")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Residuals for Each Method with the Same Input Values", fontsize=18)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(errors_richardson, label="Richardson Error")
    plt.plot(errors_steep_descent, label="Steepest Descent Error")
    #plt.semilogy(errors_richardson, label = "Richardson Error")
    #plt.semilogy(errors_steep_descent, label = "Steepest Descent Error")
    plt.axhline(y=bound_rich_steep, color="r", linestyle="--", label = "Bound of Error")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Error")
    #plt.ylabel("Relative Error (log scale)")
    plt.title("Relative Error for Each Method with the Same Input Values")
    plt.grid(True, which="both", linestyle="--", linewidth=0.8)
    plt.legend(title="Method")
    plt.show()

    return solution_1, solution_2, solution_3, iterations_1, iterations_2, iterations_3

if __name__ == "__main__":
    while True:
        solution_1, solution_2, solution_3, iterations_1, iterations_2, iterations_3 = part_1_driver()

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("Thank you for using the Iteration Solver!")


