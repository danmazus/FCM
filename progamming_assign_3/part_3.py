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

def nonzero_elements_counter(A):
    count = 0
    for i in range(1, n):
        for k in range(i):
            if A[i, k] != 0:
                count += 1

    return count

def compressed_sparse_row(A, n):
    size = nonzero_elements_counter(A)
    aa = np.zeros(size)
    ja = np.zeros(size)
    ia = np.zeros(len(aa))

    nonzero_counter = 0
    ia[0] = 0

    for i in range(n):
        for k in range(i):
            if A[i, k] != 0:
                aa[nonzero_counter] = A[i, k]
                ja[nonzero_counter] = k
                nonzero_counter += 1
        ia[i + 1] = nonzero_counter

    return aa, ja, ia

# Test Case
n = 6
a = 5
b = 10
density = 0.7
boost_factor = 2
A = generate_spd_sparse_matrix(a, b, n, density, boost_factor)
print(f"Matrix A is: \n{A}")

aa, ja, ia = compressed_sparse_row(A, n)

print(f"Compressed sparse row A is: \n{aa}")
print(f"Column Indicies are: \n{ja}")
print(f"Range of indices are: \n{ia}")