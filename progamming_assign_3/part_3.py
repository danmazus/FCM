import numpy as np
import scipy.linalg as sp
from my_package import *

def generate_spd_sparse_matrix(a, b, n, density, boost_factor):
    # Initialize Matrix L that will be updated to a Lower Triangular Matrix
    L = np.zeros((n,n))

    for i in range(n):
        # Creates a control variable that controls the number of nonzero elements in the ith row
        nonzero = max(1, int(density * i))

        # Checks the condition for the first row or anytime density >= 1
        if nonzero > i:
            nonzero = i

        # Selects the lower triangular part in each row of the controlled size variable
        ind = np.random.choice(i, size=nonzero, replace=False)

        # Fills elements of L with random values to the positions stated in the variable 'ind'
        L[i, ind] = np.random.uniform(a, b, size=nonzero)

        # Making L Symmetric by implicitly computing the transpose
        for j in ind:
            L[j, i] = L[i, j]

    # "Initializes" our diagonal elements and creates matrix A from D + L
    D = np.diag(np.diag(L))
    A = D + L

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

# Test Case
n = 5
a = 5
b = 10
density = 0.7
boost_factor = 2
A = generate_spd_sparse_matrix(a, b, n, density, boost_factor)
print(f"Matrix A is: \n{A}")