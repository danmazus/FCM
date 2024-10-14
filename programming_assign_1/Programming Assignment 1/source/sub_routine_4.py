from random import randint
import numpy as np
import math
import time
import matplotlib.pyplot as plt

''' 
This Subroutine function UL_mult_COMB takes in 2 parameters, the combined LU matrix.
The combined matrix function takes in a Unit Lower Triangular, L, and an upper triangular, U.
The inputs of the tester are listed under the run_tests function. This is where the tester function
takes the L, U, and combined function = LU to generate random matrices every run of given
user input size when run.
'''


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

# Function definition for generating random Unit Lower Triangular Matrices
def generate_unit_lower_triangular(n):
    ULT = [[0] * n for i in range(n)]
    for i in range(n):
        for k in range(n):
            if i == k:
                ULT[i][k] = 1
            elif i < k:
                ULT[i][k] = 0
            else:
                ULT[i][k] = randint(a, b)

    return ULT

# Initializing an Upper Triangular Matrix
def generate_upper_triangular(n):
    U = [[0] * n for i in range(n)]

    for i in range(n):
        for k in range(n):
            if i <= k:
                U[i][k] = randint(a, b)
            else:
                U[i][k] = 0

    return U

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

# Defining a Numpy Functions that takes in two matrices and does the dot product
def numpy_matrix_algorithm(L, U):
    L_np = np.array(L)
    U_np = np.array(U)
    W_np = np.dot(L_np, U_np)
    return W_np

def frob_norm(M):
    sum = 0
    for i in M:
        for k in i:
            sum += k ** 2
    return math.sqrt(sum)

# n = int(input("Number of dimensions: "))
# a = int(input("Lower Bound for Random Number in Vector: "))
# b = int(input("Upper Bound for Random Number in Vector: "))
# #
# L = generate_unit_lower_triangular(n)
# print("The Lower Triangular Matrix L is: ")
# for r in L:
#     print(r)
# #
# U = generate_upper_triangular(n)
# print("The Upper Triangular Matrix U is: ")
# for r in U:
#     print(r)
# #
# LU = combine_upper_lower(L, U, n)
# print("The Combination Matrix LU is: ")
# for r in LU:
#     print(r)
# #
# #
# M = UL_mult_COMB(LU, n)
# #
# print("The resulting matrix M is: ")
# for r in M:
#     print(r)
# #
# norm = frob_norm(M)
# print("The Frobenius norm of the resulting matrix is: ", norm)


#def comb_mult(LU, n):
 #   M = [[0] * n for i in range(n)]

  #  for i in range(n):
   #     for k in range(n):
    #        if i <= k:
     #           M[i][k] = LU[i][k]

      #      if i > k:
       #         for j in range(i + 1):
        #            M[i][k] += LU[i][j] * LU[j][i]
                    #print(j)
                    #print(M[i][k])
                    #print("Break \n")
            #elif i < k:
             #   for j in range(i + 1):
              #      M[i][k] += COMB[i][j] * COMB[j][k]
            #else:
             #   pass

    #return M

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
            # Inputs used for tester
            U = generate_upper_triangular(n)
            L = generate_unit_lower_triangular(n)
            LU = combine_upper_lower(L, U, n)


            # Timing Manual Algorithm
            start_time = time.time()
            m_compute_mine = UL_mult_COMB(LU, n)
            end_time = time.time()
            execution_times_manual_for_n.append(end_time - start_time)

            # Timing NumPy Algorithm
            start_time = time.time()
            m_compute_numpy = numpy_matrix_algorithm(L, U)
            end_time = time.time()
            execution_times_numpy_for_n.append(end_time - start_time)

            # Calculating Norms
            norm_mine = frob_norm(m_compute_mine)
            norm_numpy = frob_norm(m_compute_numpy)

            # Calculate difference of Norms and Appending
            diff = np.array(m_compute_mine) - np.array(m_compute_numpy)
            norm_diff = frob_norm(diff)

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
    plt.ylabel('Frobenius Norm')
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


