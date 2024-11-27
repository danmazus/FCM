from imghdr import tests

import numpy as np
import scipy.linalg as sp
from matplotlib import pyplot as plt
from scipy.sparse.linalg import spsolve_triangular
from my_package import *

def generate_spd_sparse_matrix(a, b, n, density, boost_factor):
    # Initialize Matrix A that will be updated to a Lower Triangular Matrix
    A = np.zeros((n,n))

    for i in range(n):
        # Creates a control variable that controls the number of nonzero elements in the ith row
        nonzero = max(1, int(density * i))

        # Checks the condition for the first row or anytime density >= 1
        if nonzero > i:
            nonzero = i

        # Selects the lower triangular part in each row of the controlled size variable
        ind = np.random.choice(i, size=nonzero, replace=False)

        # Fills elements of A with random values to the positions stated in the variable 'ind'
        A[i, ind] = np.random.uniform(a, b, size=nonzero)

        # Making A Symmetric by implicitly computing the transpose
        for j in ind:
            A[j, i] = A[i, j]

    # Creating random values for our diagonal elements and then assigning them to the diagonal elements of A
    diag_values = generate_float_1D_vector_np(a, b, n)
    for i in range(n):
        A[i, i] = diag_values[i]

    # Boost the diagonal of A to ensure positive definiteness
    for i in range(n):
        # Summing the off diagonal elements to see if the sum is > the diagonal element for given row i
        off_diag = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])

        # Boosting the sum of the off diagonal element by some boost_factor
        boosted = off_diag * boost_factor

        # Setting the diagonal element to either the given diagonal element or the boosted off diagonal sum
        A[i, i] = max(A[i, i], boosted)

    # Checks to see if the Matrix is positive definite as Symmetric is already known
    eigenvalues = np.linalg.eigvals(A)
    if np.all(eigenvalues) > 0:
        print("Generated Matrix is SPD")
    else:
        print("Generated Matrix is not SPD")

    return A

# Counter function for how many nonzero elements are in a given matrix
# def nonzero_elements_counter(A):
#     # Initialize the count variable
#     count = 0
#
#     # Loop over the rows of i below the diagonal
#     for i in range(1, n):
#         # Loop over the columns up to row i (this gives us all values below the diagonal)
#         for k in range(i):
#             # Checking to see if the value is nonzero
#             if A[i, k] != 0:
#                 # if the value is nonzero, add the counter by 1
#                 count += 1
#
#     return count

# Compressed sparse row function

"""Compressed Sparse Row Storage function for storing the lower triangular part of a sparse SPD matrix"""
def compressed_sparse_row_lower_tri(A):
    """
    This function takes a sparse matrix and stores the nonzero elements below the diagonal in a compressed format

    Parameters:
        A: Sparse Matrix

    Returns:
        aa: the nonzero elements of the given matrix below the diagonal (the lower triangular)
        ja: the column indices of the nonzero elements
        ia: the amount of nonzero elements in each row
    """
    # Initializes the size of the compressed sparse row storage size by the nonzero elements determined by the nonzero elements function
    #size = nonzero_elements_counter(A)
    n = len(A)

    # aa is initialized as a list that we will append with the nonzero elements
    aa = []

    # ja is initialized that will hold the column indices for the nonzero elements
    ja = []

    # ia is initialized to hold the amount of nonzero elements in each row
    ia = np.zeros(n+1, dtype=int)

    # initialize the nonzero element counter
    nonzero_counter = 0

    # Explicitly setting the initial index/starting point
    #ia[0] = 0

    # Looping over the rows of A
    for i in range(1, n):
        # Loops over the columns of A up to the ith row (gets all elements below the diagonal)
        for k in range(i):
            # Checks if element is nonzero of not
            if A[i, k] != 0:
                # If element is nonzero, append aa with that element, ja with the column index and add one to the nonzero counter
                aa.append(A[i, k])
                ja.append(k)
                nonzero_counter += 1
        # Update the next element in ia with the current amount in the nonzero counter after completing the for loop over the columns
        ia[i + 1] = nonzero_counter

    # Converting aa and ja to numpy arrays
    aa = np.array(aa)
    ja = np.array(ja)

    return aa, ja, ia

def compressed_sparse_symmetric_mat_vec_prod(aa, ia, ja, x, D):
    # Setting n as the length of the vector x
    n = len(x)

    # Initializing the resulting matrix-vector product
    b = np.zeros(n)

    # Looping over the rows
    for i in range(n):
        b[i] += D[i] * x[i]

        # Start index for row i (ia[i] is the row indices)
        k1 = ia[i]

        # Go to the ending index for row i (ia[i + 1] - ia[i] gives how many nonzero elements)
        k2 = ia[i + 1]

        for k in range(k1, k2):
            # Column index for the current element
            col_index = ja[k]

            # Value of the current column index element
            value = aa[k]

            # Update y with the lower triangular part
            b[i] += value * x[col_index]

            # Update y with the symmetric upper triangular part (column index refers to k element in A[i, k]
            b[col_index] += value * x[i]


    return b

# A lower triangular solve function for CSR
def csr_lower_solve(aa, ia, ja, b, D):
    # Initializes n as the length of b (could also do len(D))
    n = len(b)

    # Initialize the solution vector y
    y = np.zeros(n)

    # Set the first element of y as the division of b divided by the diagonal element D
    y[0] = b[0] / D[0]

    # Loops over values below the diagonal (skips the first element in aa)
    for i in range(1, n):
        # Start index for row i
        k1 = ia[i]
        # Ending index for row i
        k2 = ia[i + 1]
        # Computes the dot product of the "matrix" and vector
        temp_sum = np.dot(aa[k1:k2], y[ja[k1:k2]])
        # Sets the equivalent y[i] value
        y[i] = (b[i] - temp_sum) / D[i]

    return y

def csr_upper_solve(aa, ia, ja, y, D):
    """
    Solve Ux = y where U is upper triangular and stored as transpose of lower triangular CSR format
    """
    n = len(y)
    x = np.zeros(n)

    x[-1] = y[-1] / D[-1]

    # Process from bottom to top
    for i in range(n-2, -1, -1):
        k1 = ia[i]
        k2 = ia[i + 1]

        temp_sum = np.dot(aa[k1:k2], x[ja[k1:k2]])
        x[i] = (y[i] - temp_sum) / D[i]

    return x

def stationary_method(aa, ia, ja, D, b, x0, x_tilde, tol, max_iter, flag):
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
    r = b - compressed_sparse_symmetric_mat_vec_prod(aa, ja, ia, x, D)

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
            r_next = b - compressed_sparse_symmetric_mat_vec_prod(aa, ja, ia, x_next, D)

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
        true_err_norm = np.linalg.norm(x_true)


        while iter_num < max_iter:
            # Lower triangular solve for P^(-1) * r_k
            z = csr_upper_solve(aa, ia, ja, r, D)

            # Compute the next x term
            x_next = x + z

            rel_err = (np.linalg.norm(x_next - x_true)) / true_err_norm
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Update next values
            x = x_next
            r = b - compressed_sparse_symmetric_mat_vec_prod(aa, ja, ia, x, D)
            iter_num += 1

        return x, iter_num, rel_err_list

    # Symmetric Gauss-Seidel
    else:
        '''Symmetric Gauss-Seidel'''
        rel_err_list = []
        true_err_norm = np.linalg.norm(x_true)
        iter_num = 0

        while iter_num < max_iter:
            # First the Lower Solve is computed first (D - L)^(-1) * r_k = z_1
            z_1 = csr_lower_solve(aa, ia, ja, r, D)

            # Scale of the diagonal D * (D - L)^(-1) * r_k = z_2
            z_2 = D * z_1

            # Upper Solve is finally computed (D - U)^(-1) * z_2 = z_3
            z_3 = csr_lower_solve(aa, ia, ja, z_2, D)

            # Computing the next x
            x = x + z_3

            # Compute Relative error ||x_k - x|| / ||x||
            rel_err = (np.linalg.norm(x - x_true)) / true_err_norm
            rel_err_list.append(rel_err)

            # Checking if Relative Error is below Tolerance Level, if so return
            if rel_err < tol:
                return x, iter_num + 1, rel_err_list

            # Correcting variables for next iteration
            r = b - compressed_sparse_symmetric_mat_vec_prod(aa, ja, ia, x, D)
            iter_num += 1

        return x, iter_num, rel_err_list



#Test Case 1: Generate SPD Matrix and Check its Structure
n = 5
a = 5
c = 10
density = 0.5
boost_factor = 3

# Generate a sparse positive definite matrix
A = generate_spd_sparse_matrix(a, c, n, density, boost_factor)
print("Generated Sparse Positive Definite Matrix A:")
print(A)

# Extract and print the diagonal elements of A
D = np.diag(A)
print("\nDiagonal elements of A:")
print(D)

# Check compressed sparse row format of lower triangular part of A
aa, ja, ia = compressed_sparse_row_lower_tri(A)
print("\nCompressed Sparse Row (CSR) of Lower Triangular A:")
print("Values (aa):", aa)
print("Column indices (ja):", ja)
print("Row pointers (ia):", ia)

# Test Case 2: Matrix-Vector Multiplication (Compressed Sparse Row vs Dense)
x = np.ones(n)  # Test vector

# Matrix-vector product using compressed sparse row format
b_true = compressed_sparse_symmetric_mat_vec_prod(aa, ia, ja, x, D)
print("\nMatrix-vector product (Compressed Sparse Row) b_true:")
print(b_true)

# Matrix-vector product using dense matrix for comparison
b_tilde = np.dot(A, x)
print("\nMatrix-vector product (Dense Matrix) b_tilde:")
print(b_tilde)

# Check the difference between the two results
print("\nNorm of the difference between b_true and b_tilde:")
print(np.linalg.norm(b_true - b_tilde))

# Test Case 3: Lower Triangular Solve (Compressed Sparse Row vs Dense)
y_comp = csr_lower_solve(aa, ia, ja, b_true, D)
print("\nSolution of Lower Triangular System (Compressed Sparse Row) y_comp:")
print(y_comp)

L_dense = np.tril(A)  # Dense lower triangular matrix
y_dense = np.linalg.solve(L_dense, b_tilde)
print("\nSolution of Lower Triangular System (Dense Matrix) y_dense:")
print(y_dense)

# Check the difference between the two solutions
print("\nNorm of the difference between y_comp and y_dense:")
print(np.linalg.norm(y_comp - y_dense))

# Test Case 4: Upper Triangular Solve (Compressed Sparse Row vs Dense)
x_tilde = csr_upper_solve(aa, ia, ja, y_comp, D)
print("\nSolution of Upper Triangular System (Compressed Sparse Row) x_tilde:")
print(x_tilde)

U_dense = np.triu(A)  # Dense upper triangular matrix
x_dense = np.linalg.solve(U_dense, y_dense)
print("\nSolution of Upper Triangular System (Dense Matrix) x_dense:")
print(x_dense)

# Check the difference between the two solutions
print("\nDifference between x_tilde and x_dense:")
print(np.abs(x_tilde - x_dense))

print("\nDifference norm between x_tilde and x_dense")
print(np.linalg.norm(x_tilde - x_dense))

x0 = np.random.uniform(a, c, n)

# Compute Stationary Method
solutions_jac, iterations_jac, rel_error_list_jac = stationary_method(aa, ia, ja, D, b_tilde, x0, x, 1e-6, 1000, 1)
print(f"Iterations for Convergence Jacobi: {iterations_jac}")

solutions_gs, iterations_gs, rel_error_list_gs = stationary_method(aa, ia, ja, D, b_tilde, x0, x, 1e-6, 1000, 2)
print(f"Iterations for Convergence GS: {iterations_gs}")

solutions_sgs, iterations_sgs, rel_error_list_sgs = stationary_method(aa, ia, ja, D, b_tilde, x0, x, 1e-6, 1000, 3)
print(f"Iterations for Convergence SGS: {iterations_sgs}")




# # User input function
# def get_user_inputs():
#     """
#     Get problem parameters from user input.
#
#     Prompts user for:
#     - Matrix dimensions (n rows x n columns)
#     - Matrix to be passed (A0 through A9)
#     - Range for random values for solution vectors (smin, smax)
#     - Number of initial guess vectors (g)
#     - Range of for random values for initial guess vectors (ig_min_value, ig_max_value)
#     - Tolerance level for convergence to happen (tol)
#     - Maximum number of iterations before stopping (max_iter)
#     - Debug output preference
#     """
#
#     print("\nSparse Matrices Construction")
#     print("----------------------------------")
#
#     while True:
#         try:
#             n = int(input("Enter dimensions for vectors (n) [default=10]: ") or "10")
#
#             n_min = int(input("Minimum Dimension to be run:  [default=10]: ") or "10")
#             n_max = int(input("Maximum Dimension to be run: [default=100]: ") or "100")
#
#             density = float(input("Enter density for Sparsity (Must be between 0 and 1 [default=0.5]: ") or "0.5")
#
#             boost_factor = float(input("Enter boost factor for diagonal elements [default=2]: ") or "2")
#
#             # Random values for solution vectors to be used
#             print("\nSet range for random values of Matrices:")
#             smin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
#             smax = float(input("Enter maximum value (default=10.0): ") or "10.0")
#
#             # Checking condition to make sure input is valid
#             if smin > smax:
#                 print("Error: Minimum value must be less than maximum value")
#                 continue
#
#             # Choosing how many initial guess vectors will be taken
#             print("\nChoose How Many Solution Vectors:")
#             g = int(input("Enter number of Solution Vectors (default=10): ") or "10")
#
#             print("\nSet range for random values of Solution and Initial Guess Vectors:")
#             dmin = float(input("Enter minimum value (default=-10.0): ") or "-10.0")
#             dmax = float(input("Enter maximum value (default=10.0): ") or "10.0")
#
#             if dmin > dmax:
#                 print("Error: Minimum value must be less than maximum value")
#                 continue
#
#             print("\nChoose How Many Initial Guess Vectors for each Solution Vector:")
#             num_x0 = int(input("Enter number of Initial Guess Vectors (default=10): ") or "10")
#
#             # Tolerance level for convergence
#             print("\nSet Tolerance Level for Convergence to Hit:")
#             tol = float(input("Enter tolerance (default=1e-6): ") or "1e-6")
#
#             # Max number of iterations
#             print("\nSet Maximum Number of Iterations to be Ran:")
#             max_iter = int(input("Enter maximum number of iterations (default=1000): ") or "1000")
#
#             # Enable debug input
#             debug = input("\nEnable debug output? (y/n) (default=n): ").lower().startswith('y')
#
#             ini = input("\nRun initial case for true solution and initial guess vector? (y/n) (default=n): ").lower().startswith('y')
#             # If 'n', skip asking for method flag and set defaults
#             if not ini:
#                 # Set default values for these variables
#                 flag = 1  # Default method, for example Jacobi
#                 return n, n_min, n_max, density, boost_factor, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag
#
#             # Selecting which method to run for the selected matrix
#             print("\nChoose which method to run:")
#             print("1. Jacobi Method")
#             print("2. Forward Gauss-Seidel Method")
#             print("3. Backward Gauss-Seidel Method")
#             print("4. Symmetric Gauss-Seidel Method")
#             flag = int(input("Enter Method: "))
#
#             # Ensuring a correct value for method is chosen
#             if flag not in [1, 2, 3, 4]:
#                 print("Error: Invalid method")
#                 continue
#
#             return n, n_min, n_max, density, boost_factor, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag
#
#         except ValueError:
#             print("Error: Please enter valid numbers")
#
# def part_3_driver(n, n_min, n_max, density, boost_factor, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag):
#     results = []
#
#     for nums in range(n_min, n_max + 1):
#         matrices = []
#
#         for i in range(10):
#             A = generate_spd_sparse_matrix(smin, smax, nums, density, boost_factor)
#             matrices.append(A)
#
#         selected_matrices = np.random.choice(matrices, 3, replace=False)
#
#         for tests_index, A in enumerate(selected_matrices):
#             iterations_dict = {'Jacobi': [], 'Gauss-Seidel': [], 'Symmetric Gauss-Seidel': []}
#             D = np.diag(A)
#             aa, ja, ia = compressed_sparse_row_lower_tri(A)
#
#             for i in range(g):
#                 for j in range(num_x0):
#                     x_tilde = np.random.uniform(dmin, dmax, nums)
#                     b_tilde = compressed_sparse_symmetric_mat_vec_prod(aa, ia, ja, x_tilde, D)
#                     x0 = np.random.uniform(dmin, dmax, nums)
#
#                     solution, iterations, rel_error_list = stationary_method(aa, ia, ja, D, b_tilde, x0, x_tilde,
#                                                                                      tol, max_iter, 1)
#                     iterations_dict['Jacobi'].append(iterations)
#
#                     solution, iterations, rel_error_list = stationary_method(aa, ia, ja, D, b_tilde, x0, x_tilde,
#                                                                                   tol, max_iter, 2)
#                     iterations_dict['Gauss-Seidel'].append(iterations)
#
#                     solution, iterations, rel_error_list = stationary_method(aa, ia, ja, D, b_tilde, x0,
#                                                                                      x_tilde, tol, max_iter, 3)
#                     iterations_dict['Symmetric Gauss-Seidel'].append(iterations)
#
#             fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#
#             axs[0].plot(iterations_dict['Jacobi'], label='Jacobi', marker='o', linestyle='-', color='b')
#             axs[0].set_title('Jacobi Method')
#             axs[0].set_xlabel('Test Index')
#             axs[0].set_ylabel('Iterations')
#
#             axs[1].plot(iterations_dict['Gauss-Seidel'], label='Gauss-Seidel', marker='x', linestyle='--', color='g')
#             axs[1].set_title('Gauss-Seidel Method')
#             axs[1].set_xlabel('Test Index')
#             axs[1].set_ylabel('Iterations')
#
#             axs[2].plot(iterations_dict['Symmetric Gauss-Seidel'], label='Symmetric Gauss-Seidel', marker='s',
#                         linestyle='-.', color='r')
#             axs[2].set_title('Symmetric Gauss-Seidel Method')
#             axs[2].set_xlabel('Test Index')
#             axs[2].set_ylabel('Iterations')
#
#             for ax in axs:
#                 ax.legend()
#                 ax.grid(True)
#
#             plt.tight_layout()
#             plt.show()
#
#             results.append({
#                 'matrix_size': nums,
#                 'density': density,
#                 'test_id': tests_index + 1,
#                 'iterations_dict': iterations_dict
#             })
#
#     return results
#
#
#
# # Main function
# if __name__ == "__main__":
#     while True:
#         inputs = get_user_inputs()
#         n, n_min, n_max, density, boost_factor, smin, smax, g, dmin, dmax, num_x0, tol, max_iter, debug, ini, flag= inputs
#
#         if not ini:
#             results = part_3_driver(n, n_min, n_max, density, boost_factor, smin, smax, g, dmin, dmax, num_x0, tol, max_iter,
#                           debug, ini, flag)
#
#         user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
#         if user_input != 'y':
#             break
#
#         print("Thank you for using the Solver!")
