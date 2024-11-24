import numpy as np
import scipy.linalg as sp
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

# Compressed Sparse Row Storage function for storing the lower triangular part of a sparse SPD matrix
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
    ia[0] = 0

    # Looping over the rows of A
    for i in range(n):
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

def compressed_sparse_mat_vec_prod(aa, ia, ja, x):
    # Setting n as the length of the vector x
    n = len(x)

    # Initializing the resulting matrix-vector product
    y = np.zeros(n)

    # Looping over the rows
    for i in range(n):
        # Start index for row i (ia[i] is the row indices)
        k1 = ia[i]
        # Go to the ending index for row i (ia[i + 1] - ia[i] gives how many nonzero elements)
        k2 = ia[i + 1]
        # Compute the dot product from aa[ia[i] : ia[i+1]] and the corresponding column indices for x (x[ja[ia[i] : ia[i+1]]])
        y[i] = np.dot(aa[k1:k2], x[ja[k1:k2]])

    return y

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

# Test Case
n = 6
a = 5
b = 10
density = 0.7
boost_factor = 2
A = generate_spd_sparse_matrix(a, b, n, density, boost_factor)
print(f"Matrix A is: \n{A}")

D = np.diag(A)
print(f"Diagonal elements of A are: \n{D}")

aa, ja, ia = compressed_sparse_row_lower_tri(A)

print(f"Compressed sparse row A is: \n{aa}")
print(f"Column Indicies are: \n{ja}")
print(f"Range of indices are: \n{ia}")

x = np.ones(n)
print(f"Vector x is: {x}")

y = compressed_sparse_mat_vec_prod(aa, ia, ja, x)
print(f"Vector y is: \n{y}")
D_x = D * x
print(f"Vector D_x is: \n{D_x}")
y_final = y + D_x
print(f"Vector y_final is: \n{y_final}")

b_tilde = np.dot(A, x)
print(f"Vector b_tilde is: \n{b_tilde}")

x_tilde = csr_lower_solve(aa, ia, ja, b_tilde, D)
print(f"Vector x_tilde is: \n{x_tilde}")

D_L = np.tril(A)
print(f"Lower triangular matrix D_L is: \n{D_L}")

x_back = np.dot(D_L, x_tilde)
print(f"Vector x_back is: \n{x_back}")