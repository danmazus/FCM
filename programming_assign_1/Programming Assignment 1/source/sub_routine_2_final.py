from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np
import time

'''
This subroutine is for compressed row vector of a unit lower triangular matrix.
Our function for computation is the first function in the list of functions which takes
3 parameters, crv, the compressed row vector, a vector v, and n. We do the same generations of
unit lower triangular matrices as in subroutine 4 and 1. We use the compressed_row function
which takes in L, n, and index = 0 for indexing purposes. This will output our 1D array, crv
that stores the elements below the diagonal. The tester remains the same with inputs and where to find and how
to use.
'''

#### FUNCTIONS ####
# Function Definition for Unit Lower Triangular and Vector Multiplication stored in Compressed Columns
def Lv_mult_cr(crv, v, n):
    # Initialization for Resulting Vector
    m = [0] * n
    # Index to track position in CRV
    index = 0

    for i in range(n):
        for k in range(i):
            # This is all non-diagonal elements below the diagonal in the given row i
            m[i] += crv[index] * v[k]
            index += 1
        # Adding the diagonal element -- this gave me trouble to figure out
        m[i] += v[i]

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

# Function definition for Compressed Row Vector
def compressed_row(L, n, index = 0):
    crv_size = ((n * (n - 1)) // 2)
    crv = [0] * crv_size
    for i in range(n):
        for k in range(i):
            crv[index] = L[i][k]
            index += 1

    return crv

# Function definition for generating 1D random vectors
def generate_1D_vector(n):
    return [randint(a, b) for i in range(n)]

# Function defining the Euclidean Norm
def vec_2_norm(v):
    return math.sqrt(sum(x**2 for x in v))

# Function definition for Using NumPy's Matrix, Vector Multiplication for ULT
def numpy_algorithm(L, v):
    L_np = np.array(L)
    v_np = np.array(v)
    w_np = np.dot(L_np, v_np)
    return w_np

# L = generate_unit_lower_triangular(n)
# print("The Unit Lower Triangular Matrix is: ")
# for r in L:
#     print(r)
#
# crv = compressed_row(L, n)
# print("The Compressed Row Storage for ULT is: ", crv)
#
# v = generate_1D_vector(n)
# print("The 1D Vector v is: ", v)
#
# m = Lv_mult_cr(crv, v, n)
# print("The resulting 1D array m is: ", m)
#
# norm = vec_2_norm(m)
# print("The resulting 2 Norm for m is: ", norm)
#
# npmatrix = numpy_algorithm(L, v)
# print("NPMatrix is: ")
# for r in npmatrix:
#     print(r)
#
# npnorm = vec_2_norm(npmatrix)
# print("The resulting 2 Norm for npnorm is: ", npnorm)

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
            L = generate_unit_lower_triangular(n)
            v = generate_1D_vector(n)
            crv = compressed_row(L, n)

            # Timing Manual Algorithm
            start_time = time.time()
            m_compute_mine = Lv_mult_cr(crv, v, n)
            end_time = time.time()
            execution_times_manual_for_n.append(end_time - start_time)

            # Timing NumPy Algorithm
            start_time = time.time()
            m_compute_numpy = numpy_algorithm(L, v)
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
