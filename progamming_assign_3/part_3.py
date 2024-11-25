import numpy as np
import scipy.linalg as sp
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

    # Process from bottom to top
    for j in range(n - 1, -1, -1):
        sum_val = y[j]

        start = ia[j]
        end = ia[j+1]

        for index in range(start, end):
            i = ja[index]
            if i < j:
                sum_val -= aa[index] * x[i]

        x[j] = sum_val / D[j]

    return x


# Test Case 1: Generate SPD Matrix and Check its Structure
n = 5
a = 5
c = 10
density = 0.7
boost_factor = 2

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






