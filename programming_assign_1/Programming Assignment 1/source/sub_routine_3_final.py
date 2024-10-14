from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np
import time

'''
This Subroutine is for a banded matrix with 2 sub-diagonals as the bands. Our function
Lv_mult_banded takes in 3 parameters, W, the matrix that stores the sub-diagonals, a vector v,
and n. We generate v from the function generate_1D_vector and generate a banded matrix under banded_matrix.
This banded matrix, B, is then the input for the storage function, stored_banded, where this will
output W. The tester remains the same as the other subroutines for where to put inputs and the explanation.
'''


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

# Function definition for generating 1D random vectors
def generate_1D_vector(n):
    return [randint(a, b)] * n

# Function definition for generating a 2 bandwidth banded unit matrix
def banded_matrix(n):
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

    return W

#B = banded_matrix(n)
#W = stored_banded(B, n)
#v = generate_1D_vector(n)

#m = Lv_mult_banded(W, v, n)
#print("The resulting vector m is: ", m)

#### TESTER ####

# Function definition for Using NumPy's Matrix, Vector Multiplication for ULT
def numpy_algorithm(B, v):
    L_np = np.array(B)
    v_np = np.array(v)
    w_np = np.dot(L_np, v_np)
    return w_np

# Function definition for taking the Euclidean Norm of a vector
def vec_2_norm(v):
    return math.sqrt(sum(x**2 for x in v))

# def run_tests(dim_min, dim_max, num_tests, a, b):
#     # Putting Dimensions into List
#     dimensions = list(range(dim_min, dim_max + 1))
#     # Initializing the different lists to store runs
#     manual_norms = []
#     numpy_norms = []
#     diff_norms = []
#
#     # Initializing the lists to store each n run which will be appended each run
#     for n in dimensions:
#         manual_norms_for_n = []
#         numpy_norms_for_n = []
#         diff_norms_for_n = []
#
#         for i in range(num_tests):
#             # Setting Values Needed for computation with built functions
#             B = banded_matrix(n)
#             v = generate_1D_vector(n)
#             W = stored_banded(B, n)
#
#             # Setting Self-Computed Function
#             m_compute_mine = Lv_mult_banded(W, v, n)
#
#             # Setting NumPy's Function for Comparison to Calculate True
#             m_compute_numpy = numpy_algorithm(B, v)
#
#             # Calculating 2-Norms both custom and numpy norms to ensure correctness
#             norm_mine = vec_2_norm(m_compute_mine)
#             norm_numpy = vec_2_norm(m_compute_numpy)
#
#             # Calculating the difference of the norms and taking norm of differences
#             diff = np.array(m_compute_mine) - np.array(m_compute_numpy)
#             norm_diff = vec_2_norm(diff)
#
#             # Appending list each run to store
#             manual_norms_for_n.append(norm_mine)
#             numpy_norms_for_n.append(norm_numpy)
#             diff_norms_for_n.append(norm_diff)
#
#         # Appending Original list with the mean for each dimension
#         manual_norms.append(np.mean(manual_norms_for_n))
#         numpy_norms.append(np.mean(numpy_norms_for_n))
#         diff_norms.append(np.mean(diff_norms_for_n))
#
#     # Plotting the Comparison and Differences with Dimension
#     plt.figure(figsize=(12, 6))
#
#     plt.plot(dimensions, manual_norms, label = 'Manually Computed Norms', marker = 'o')
#     plt.plot(dimensions, numpy_norms, label = 'Numpy Norms', marker = 'x')
#     plt.plot(dimensions, diff_norms, label = 'Difference Norms', marker = 'v')
#
#     plt.xlabel('Matrix Dimension (n)')
#     plt.ylabel('Euclidean Norm')
#     plt.title('Comparison of Norms for Custom Matrix-Vector Multiplication and True Computation by NumPy')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


'''#### TESTER #### '''

def run_tests(dim_min, dim_max, num_tests, a, b):
    dimensions = list(range(dim_min, dim_max + 1))
    manual_norms = []
    numpy_norms = []
    diff_norms = []
    execution_times_manual = []
    execution_times_numpy = []

    for n in dimensions:
        manual_norms_for_n = []
        numpy_norms_for_n = []
        diff_norms_for_n = []
        execution_times_manual_for_n = []
        execution_times_numpy_for_n = []

        for i in range(num_tests):
            B = banded_matrix(n)
            v = generate_1D_vector(n)
            W = stored_banded(B, n)

            # Timing Manual Algorithm
            start_time = time.time()
            m_compute_mine = Lv_mult_banded(W, v, n)
            end_time = time.time()
            execution_times_manual_for_n.append(end_time - start_time)

            # Timing NumPy Algorithm
            start_time = time.time()
            m_compute_numpy = numpy_algorithm(B, v)
            end_time = time.time()
            execution_times_numpy_for_n.append(end_time - start_time)

            # Calculating Norms
            norm_mine = vec_2_norm(m_compute_mine)
            norm_numpy = vec_2_norm(m_compute_numpy)

            # Calculate difference of Norms and Appending
            diff = np.array(m_compute_mine) - np.array(m_compute_numpy)
            norm_diff = vec_2_norm(diff)

            manual_norms_for_n.append(norm_mine)
            numpy_norms_for_n.append(norm_numpy)
            diff_norms_for_n.append(norm_diff)

        # Averages
        manual_norms.append(np.mean(manual_norms_for_n))
        numpy_norms.append(np.mean(numpy_norms_for_n))
        diff_norms.append(np.mean(diff_norms_for_n))
        execution_times_manual.append(np.mean(execution_times_manual_for_n))
        execution_times_numpy.append(np.mean(execution_times_numpy_for_n))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(dimensions, manual_norms, label='Manually Computed Norms', marker='o')
    plt.plot(dimensions, numpy_norms, label='Numpy Norms', marker='x')
    plt.plot(dimensions, diff_norms, label='Difference Norms', marker='v')
    plt.xlabel('Matrix Dimension (n)')
    plt.ylabel('Euclidean Norm')
    plt.title('Manually Computed Norms vs NumPy Norms')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(dimensions, execution_times_manual, label='Manually Executed Times', marker='o', color = 'r')
    plt.plot(dimensions, execution_times_numpy, label='Numpy Executed Times', marker='x', color = 'b')
    plt.xlabel('Matrix Dimension (n)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Execution Times vs Dimensions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

'''User inputs to determine dimensions and Bounds for Random Numbers'''
dim_min = int(input("Minimum dimension: "))
dim_max = int(input("Maximum dimension: "))
num_tests = int(input("Number of tests per dimension: "))
a = int(input("Lower Bound for Random Number in Vector: "))
b = int(input("Upper Bound for Random Number in Vector: "))

run_tests(dim_min, dim_max, num_tests, a, b)
