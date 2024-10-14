# Function Definition for Unit Lower Triangular Matrix and vector product
def Lv_mult(L, v, n):
    m = [0] * n

    # Loops over rows of M
    for i in range(n):
        for k in range(n):
            m[i] += L[i][k] * v[k]

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
def generate_unit_lower_triangular(n):
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

# Function definition for generating 2D random vectors
def generate_2D_vector(n):
    return [[randint(a, b)] for i in range(n)]

# Function definition for generating 1D random vectors
def generate_1D_vector(n):
    return [randint(a, b)] * n

# Function definition for generating random Unit Upper Triangular Matrices
def generate_unit_upper_triangular(n):
    UUT = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            if i == k:
                UUT[i][k] = 1
            elif i > k:
                UUT[i][k] = 0
            else:
                UUT[i][k] = randint(a, b)

    return UUT

# Generate a random Upper Triangular Matrix
def generate_upper_triangular(n):
    UT = [[0] * n for i in range(n)]

    for i in range(n):
        for k in range(n):
            if i <= k:
                UUT[i][k] = 3
            else:
                UUT[i][k] = 0

# Function definition for Compressed Row Vector
def compressed_row(ULT, n, index = 0):
    crv_size = ((n * (n - 1)) // 2)
    crv = [0] * crv_size
    for i in range(n):
        for k in range(i):
            crv[index] = ULT[i][k]
            index += 1

    return crv

# Function definition for Using NumPy's Matrix, Vector Multiplication for ULT
def numpy_algorithm(ULT, v):
    L_np = np.array(ULT)
    v_np = np.array(v)
    w_np = np.dot(L_np, v_np)
    return w_np

# Function definition for generating a banded matrix
def banded_matrix(n):
    # Initialization for Banded Matrix
    brv = [[0] * n for i in range(n)]

    # This is now creating a matrix brv that has zeros everywhere except for the 2 sub-diagonals
    for i in range(n):
        for k in range(i + 1):
            if i - 1 == k:
                brv[i][k] = randint(a, b)  # ULT[i][k]
            elif i - 2 == k:
                brv[i][k] = randint(a, b)  # ULT[i][k]
            elif i == k:
                brv[i][k] = 1  # ULT[i][k]
            else:
                brv[i][k] = 0

    return brv

# Function definition for taking banded matrix and storing into a 2D array and arranging terms
def stored_banded(brv, n):
    vb = [[0] * (n - 1) for i in range(2)]

    for i in range(n):  # rows
        for k in range(i):  # columns
            if i - 2 == k:
                vb[0][k] = brv[i][k]
            elif i - 1 == k:
                vb[1][k] = brv[i][k]
            else:
                k += 1

    # Shifting the Zero to the front of the array to show that there is a missing value because of the bands
    for i in range(1):
        vb[i] = [0] + vb[i][:-1]

# Function definition for taking the L2 Norm (Generalized Euclidean norm for matrices)
def gen_euclidean_norm(L, n):
    return math.sqrt(sum(L[i][k] ** 2 for i in range(n) for k in range(n[0])))

# Function definition for taking the Euclidean Norm of a vector
def vec_2_norm(v):
    return math.sqrt(sum(x**2 for x in v))
