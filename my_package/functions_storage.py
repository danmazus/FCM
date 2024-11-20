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

# Computes Householder Vector 'u' and scalar 'rho'
def compute_house(v, nj):
    """
    Compute Householder vector u and scalar rho.
    This function transforms a vector 'v', input, into a form of Hv = rho * e_1
    using Parlett's approach.

    Parameters:
        v (numpy array): Vector to be transformed
        nj (int): Number of elements of 'v' to be considered

    Returns:
        u (numpy array): Householder vector
        rho (float): Scalar value for transformed vector
    """
    # Step 1: Copy first 'nj' elements of 'v' into 'w'
    w = np.copy(v[:nj])

    # Step 2: Compute mu, which is the sum of squares of elements from the second element to the nj-th element in 'v'
    mu = np.sum(v[1:nj] ** 2)

    # Step 3: Compute rho, the 2-norm of 'v'
    rho = np.sqrt(v[0]**2 + mu)

    # Step 4: Update the first element of 'w' based on the sign of 'v[0]'
    if v[0] <= 0:
        w[0] -= rho
    else:
        w[0] = -mu / (v[0] - rho)

    # Step 5: Compute the 2-norm of w
    w2norm = vec_2_norm(w)

    # Step 6: Normalize 'w' to get the Householder vector 'u'
    u = w / w2norm

    return u, rho

# Lower Triangular Solve
def solve_Lb(LU, b, n):
    """Solves the equation Ly = b (forward substitution)

    Parameters include:
    LU: a combination matrix of unit lower and upper triangular matrices stored in single 2D array
    b: the vector from the equation Ax = b
    n: the dimension of vector and matrix given by user

    Returns:
    y: the vector to use in Ux solver (Ux = y)
    """
    # Initializing vector y
    y = [0 for i in range(n)]

    # Setting the first element
    y[0] = b[0]

    for i in range(n):
        # Initializing temp_sum variable
        temp_sum = 0

        # Summing the L[i][k] * y[k]
        for k in range(i):
            # Accumulating the sum before updating
            temp_sum += LU[i][k] * y[k]
        y[i] = b[i] - temp_sum

    return y

# Upper Triangular Solve
def solve_Ux(LU, y, n):
    """Solves the Equation Ux = y with Backwards Substitution

    Parameters include:
    LU: a combination matrix of unit lower and upper triangular matrices stored in single 2D array
    y: the output vector from Lb_solver
    n: the dimension specified by user.

    Returns:
    x: the solution vector.
    """
    # Initialize x
    x = [0 for i in range(n)]

    # Starting from last row going upwards (Backward Substitution)
    for i in range(n-1, -1, -1):
        temp_sum = 0

        # Calculate the sums for k > i
        for k in range(i + 1, n):
            temp_sum += LU[i][k] * x[k]

        # Solving for x[i]
        x[i] = (y[i] - temp_sum)/LU[i][i]

    return x


### Numpy-based functions that preserve above computations ###

def generate_unit_lower_triangular_np(a, b, n):
    """Generate a unit lower triangular matrix using numpy arrays but manual filling"""
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                L[i, k] = 1
            elif i > k:
                L[i, k] = randint(a, b)
    return L


def generate_cond_unit_lower_triangular_np(a, n):
    """Generate a conditioned unit lower triangular matrix using numpy arrays"""
    L = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                L[i, k] = 1.0
            elif i > k:
                L[i, k] = np.random.uniform(-0.99999, 0.9999)
    return L


def Lv_mult_np(L, v, n):
    """Unit lower triangular matrix-vector product using numpy arrays but manual computation"""
    m = np.zeros(n)

    for i in range(n):
        for k in range(i):
            m[i] += L[i, k] * v[k]
        m[i] += v[i]  # Adding diagonal element

    return m


def Lv_mult_banded_np(W, v, n):
    """2-bandwidth banded matrix-vector product using numpy arrays"""
    m = np.zeros(n)

    for i in range(n):
        m[i] += v[i]  # Diagonal of 1 * v

        if i >= 1:  # First sub-diagonal
            m[i] += W[1, i - 1] * v[i - 1]

        if i >= 2:  # Second sub-diagonal
            m[i] += W[0, i - 1] * v[i - 2]

    return m


def generate_unit_upper_triangular_np(a, b, n):
    """Generate unit upper triangular matrix using numpy arrays"""
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i == k:
                U[i, k] = 1
            elif i < k:
                U[i, k] = randint(a, b)
    return U


def generate_upper_triangular_np(a, b, n):
    """Generate upper triangular matrix using numpy arrays"""
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(i, n):  # Only iterate over upper triangle
            U[i, k] = randint(a, b)
    return U


def generate_nonsingular_upper_triangular_np(a, b, n, ratio):
    """Generate nonsingular upper triangular matrix using numpy arrays"""
    U = np.zeros((n, n))

    for i in range(n):
        # Set diagonal elements
        U[i, i] = np.random.uniform(a, b)
        while -1 < U[i, i] < 1:
            U[i, i] = np.random.uniform(a, b)

        # Set off-diagonal elements
        for k in range(i + 1, n):
            max_value = abs(U[i, i]) * ratio
            U[i, k] = random.uniform(-max_value, max_value)

    return U


def UL_mult_COMB_np(LU, n):
    """Unit lower triangular and upper triangular matrix product using numpy arrays"""
    M = np.zeros((n, n))

    for i in range(n):
        for k in range(n):
            if i == k:
                # Diagonal case
                M[i, k] += LU[i, k]
                for j in range(i):
                    M[i, k] += LU[i, j] * LU[j, k]
            elif i > k:
                # Lower triangular part
                for j in range(k + 1):
                    M[i, k] += LU[i, j] * LU[j, k]
            else:
                # Upper triangular part
                M[i, k] += LU[i, k]
                for j in range(i):
                    M[i, k] += LU[i, j] * LU[j, k]
    return M


def combine_upper_lower_np(L, U, n):
    """Combine lower and upper triangular matrices using numpy arrays"""
    LU = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            if i > k:
                LU[i, k] = L[i, k]  # Lower triangular part
            else:
                LU[i, k] = U[i, k]  # Upper triangular and diagonal
    return LU


def compute_house_np(v, nj):
    """Compute Householder vector and scalar using numpy arrays but manual computation"""
    w = np.copy(v[:nj])
    mu = np.sum(v[1:nj] ** 2)  # Sum of squares of elements after first
    rho = np.sqrt(v[0] ** 2 + mu)

    # Update first element based on sign
    if v[0] <= 0:
        w[0] -= rho
    else:
        w[0] = -mu / (v[0] - rho)

    w2norm = np.sqrt(np.sum(w ** 2))  # Manual 2-norm computation
    u = w / w2norm  # Normalize

    return u, rho


def gen_euclidean_norm_np(L):
    """Generalized Euclidean norm for matrices using numpy arrays but manual computation"""
    return np.sqrt(np.sum(L ** 2))


def vec_2_norm_np(v):
    """Euclidean norm of a vector using numpy arrays but manual computation"""
    return np.sqrt(np.sum(v ** 2))


def frob_norm_np(M):
    """Frobenius norm using numpy arrays but manual computation"""
    return np.sqrt(np.sum(M ** 2))

def generate_float_1D_vector_np(a, b, n):
    return np.random.uniform(a, b, n)

def generate_float_1D_uniform_vector_np(a, b, n):
    Lambda = np.zeros(n)

    Lambda[0] = a
    Lambda[-1] = b

    Lambda[1:-1] = np.random.uniform(a, b, n-2)

    return Lambda

def generate_float_normal_vector_np(a, b, n):
    Lambda = np.zeros(n)

    Lambda[0] = a
    Lambda[-1] = b

    mean = (a + b) / 2
    stan_dev = (a + b) / 6  # Put values within 3 standard deviations

    Lambda[1:-1] = np.clip(np.random.normal(mean, stan_dev, n - 2), a, b)

    return Lambda
