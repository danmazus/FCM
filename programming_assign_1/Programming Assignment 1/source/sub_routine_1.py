import math
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import time

'''
This subroutine takes a unit lower triangular matrix and multiplies it by a vector v.
We generate the unit lower triangular matrix by filling in 1's on the diagonal and taking
random integers below the diagonal and the rest is filled with 0s. Our computation function,
Lv_mult takes in 3 parameters: L, v, and n. We generate random vectors by the generate_1D_vector
function. We also have created a norm calculator function called vec_2_norm that is used within the tester.
When running the routines, the command window will ask for user inputs to determine how many runs and how many
dimensions to run over. This is where the tester function comes into play. The tester function
will take inputs designated under the nested for loop and has several parameters that are filled by user inputs.
This tester will run through the different dimensions given and store and print out two graphs that
are filled by the norms and execution times per dimension. The printed norms and execution times are averages over
how many runs specified.
'''


# Function Definition for Unit Lower Triangular Matrix and vector product
def Lv_mult(L, v, n):
    m = [0] * n

    # Loop over each row
    for i in range(n):
        # Loop over elements below the diagonal
        for k in range(i):
                m[i] += L[i][k] * v[k]

        # Adding the Diagonal Element
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

# Function definition for generating 1D random vectors
def generate_1D_vector(n):
    return [randint(a, b) for i in range(n)]

#L = generate_unit_lower_triangular(n)
#print("The Unit Triangular Matrix is: ")
#for r in L:
 #   print(r)

#v = generate_1D_vector(n)
#print("The 1D Vector v is: ", v)

#m = Lv_mult(L, v, n)
#print("The resulting 1D Array M is: ", m)

def vec_2_norm(v):
    return math.sqrt(sum(x**2 for x in v))

# Function definition for Using NumPy's Matrix, Vector Multiplication for ULT
def numpy_algorithm(L, v):
    L_np = np.array(L)
    v_np = np.array(v)
    w_np = np.dot(L_np, v_np)
    return w_np

#### TESTER ####

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

            # Timing Manual Algorithm
            start_time = time.time()
            m_compute_mine = Lv_mult(L, v, n)
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
    plt.title('Comparison of Norms for Computed vs True')
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

# User inputs to determine dimensions and Bounds for Random Numbers
dim_min = int(input("Minimum dimension: "))
dim_max = int(input("Maximum dimension: "))
num_tests = int(input("Number of tests per dimension: "))
a = int(input("Lower Bound for Random Number in Vector: "))
b = int(input("Upper Bound for Random Number in Vector: "))



run_tests(dim_min, dim_max, num_tests, a, b)