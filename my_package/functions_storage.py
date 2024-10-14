import random
from random import randint
import math
import numpy as np

# Function Definition for Unit Lower Triangular Matrix and vector product
def Lv_mult(L, v, n):
    m = [0] * n

    # Loops over each row
    for i in range(n):
        # Loop over elements below the diagonal
        for k in range(i):
            m[i] += L[i][k] * v[k]

        # Adding the Diagonal Element
        m[i] += v[i]

    return m

# Function Definition for Unit Lower Triangular and Vector Multiplication stored in Compressed Columns
def Lv_mult_cr(crv, v, n):
    # Initialization for Resulting Vector
    m = [0] * n
    # Index to track position in CRV
    index = 0

    # Loops over rows of M
    for i in range(n):
        for k in range(i):
            # This is all non-diagonal elements below the diagonal in the given row i
            m[i] = crv[index] * v[k]
            index += 1
        # Adding the diagonal element -- this gave me trouble to figure out
        m[i] += v[i]

    return m

# Function definition for a 2-bandwidth banded matrix and vector product
def Lv_mult_banded(W, v, n):
    # Initialization for Resulting Vector
    m = [0] * n

    # Loops over rows of m
    for i in range(n):

        # Multiplying the diagonal of 1 * v for each i
        m[i] += v[i]

        # Checking to see if 1st sub-diagonal exists
        if i >= 1:
            m[i] += W[1][i - 1] * v[i - 1]

        # Checking to see if 2nd sub-diagonal exists
        if i >= 2:
            m[i] += W[0][i - 1] * v[i - 2]

    return m

# Function definition for generating random Unit Lower Triangular Matrices
def generate_unit_lower_triangular(a, b, n):
    L = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            if i == k:
                L[i][k] = 1
            elif i < k:
                L[i][k] = 0
            else:
                L[i][k] = randint(a, b)

    return L

# Function definition for generating random nonsingular and conditioned Unit Lower Triangular Matrices
def generate_cond_unit_lower_triangular(a, n):
    """

    """
    L = [[0.0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            if i == k:
                L[i][k] = 1.0
            elif i < k:
                L[i][k] = 0.0
            else:
                L[i][k] = np.random.uniform(-0.99999, 0.9999)

    return L

# Function definition for generating 2D random vectors
def generate_2D_vector(a, b, n):
    return [[randint(a, b)] for i in range(n)]

# Function definition for generating 1D random vectors
def generate_1D_vector(a, b, n):
    return [randint(a, b) for i in range(n)]

# Function definition for generating 1D random vectors with floating point integers
def generate_float_1D_vector(a, b, n):
    return [np.random.uniform(a, b) for i in range(n)]

# Function definition for generating random Unit Upper Triangular Matrices
def generate_unit_upper_triangular(a, b, n):
    U = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            if i == k:
                U[i][k] = 1
            elif i > k:
                U[i][k] = 0
            else:
                U[i][k] = randint(a, b)

    return U

# Generate a random Upper Triangular Matrix
def generate_upper_triangular(a, b, n):
    U = [[0] * n for i in range(n)]

    for i in range(n):
        for k in range(n):
            if i <= k:
                U[i][k] = randint(a, b)
            else:
                U[i][k] = 0

    return U

# Generate a random nonsingular Upper Triangular Matrix
def generate_nonsingular_upper_triangular(a, b, n, ratio):
    """
    Description:
        Generates non-singular upper triangular matrix with conditioned off-diagonal elements

    Parameters:
        a: lower bound
        b: upper bound
        n: number of dimensions
        ratio: Constraining off-diagonal elements to be within a ratio/range of the diagonal elements

    Returns:
        U: upper triangular matrix that is non-singular and should be conditioned based off ratio set
    """
    U = [[0] * n for i in range(n)]

    for i in range(n):
        # Setting diagonal elements where the values cannot go between -1 and 1 to preserve condition
        U[i][i] = np.random.uniform(a, b)
        while -1 < U[i][i] < 1:
            U[i][i] = np.random.uniform(a, b)

        # Constraining off-diagonal elements to be within a ratio of the magnitude of the diagonal elements
        for k in range(i + 1, n):
            max_value = abs(U[i][i]) * ratio
            U[i][k] = random.uniform(-max_value, max_value)

    return U

# Function definition for Compressed Row Vector
def compressed_row(L, n, index = 0):
    crv_size = ((n * (n - 1)) // 2)
    crv = [0] * crv_size
    for i in range(n):
        for k in range(i):
            crv[index] = L[i][k]
            index += 1

    return crv

# Function definition for Using NumPy's Matrix, Vector Multiplication for ULT
def numpy_algorithm(L, v):
    L_np = np.array(L)
    v_np = np.array(v)
    w_np = np.dot(L_np, v_np)
    return w_np

# Defining a Numpy Functions that takes in two matrices and does the dot product
def numpy_matrix_algorithm(L, U):
    L_np = np.array(L)
    U_np = np.array(U)
    W_np = np.dot(L_np, U_np)
    return W_np

# Function definition for generating a banded matrix
def banded_matrix(a, b, n):
    # Initialization for Banded Matrix
    B = [[0] * n for i in range(n)]

    # This is now creating a matrix B that has zeros everywhere except for the 2 sub-diagonals
    for i in range(n):
        for k in range(i + 1):
            if i - 1 == k:
                B[i][k] = randint(a, b)
            elif i - 2 == k:
                B[i][k] = randint(a, b)
            elif i == k:
                B[i][k] = 1
            else:
                B[i][k] = 0

    return B

# Function definition for taking banded matrix and storing into a 2D array and arranging terms
def stored_banded(B, n):
    W = [[0] * (n - 1) for i in range(2)]

    for i in range(n):  # rows
        for k in range(i):  # columns
            if i - 2 == k:
                W[0][k] = B[i][k]
            elif i - 1 == k:
                W[1][k] = B[i][k]
            else:
                k += 1

    # Shifting the Zero to the front of the array to show that there is a missing value because of the bands
    for i in range(1):
        W[i] = [0] + W[i][:-1]

# Function definition for taking the L2 Norm (Generalized Euclidean norm for matrices)
def gen_euclidean_norm(L, n):
    return math.sqrt(sum(L[i][k] ** 2 for i in range(n) for k in range(n[0])))

# Function definition for taking the Euclidean Norm of a vector
def vec_2_norm(v):
    return math.sqrt(sum(x**2 for x in v))

# Function definition for taking the Frobeinan Norm
def frob_norm(M):
    sum = 0
    for i in M:
        for k in i:
            sum += k ** 2
    return math.sqrt(sum)

# Function definition for a Unit Lower Triangular and Upper Triangular Matrix Product
def UL_mult_COMB(LU, n):
    M = [[0] * n for i in range(n)]
    #for i in range(n):
     #   for k in range(i, n):
      #      M[i][k] = LU[i][k]

    #or i in range(n):
       # for k in range(n):
          #  upper_limit = min(i, k + 1)
          #  for j in range(upper_limit):
             #   M[i][k] += LU[i][j] * LU[j][k]

    for i in range(n):
        for k in range(n):
            if i == k:
                # Diagonal Case
                M[i][k] += LU[i][k]
                for j in range(i):
                    M[i][k] += LU[i][j] * LU[j][k]
            elif i > k:
                # Lower Triangular Part
                for j in range(k + 1):
                    M[i][k] += LU[i][j] * LU[j][k]
            else:
                # i < k, Upper Triangular Part
                M[i][k] += LU[i][k]
                for j in range(i):
                      M[i][k] += LU[i][j] * LU[j][k]
    return M

# Combining L and U matrices taking in L and U as parameters
def combine_upper_lower(L, U, n):
    LU = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            # Pulling elements out of ULT and storing them in LU below the diagonal
            if i > k:
                LU[i][k] = L[i][k]
            # Pulling elements out of UUT and storing them in LU on and above the diagonal
            else:
                LU[i][k] = U[i][k]

    return LU
