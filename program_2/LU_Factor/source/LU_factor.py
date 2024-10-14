# Imports
import copy
import my_package
import numpy as np
import matplotlib.pyplot as plt
import random
from program_2.LU_Factor.source.Lv_solver import solve_Lb
from program_2.LU_Factor.source.Uv_solver import solve_Ux


# LU Factorization Function taking in matrix A, composed of L and U or a random dense matrix, n (the dimensions), and
# routine (indicating no partial pivoting or including partial pivoting denoted by 1 and 2
def LU_factor(A, n, routine):

    """
    Description:
        Performs LU Factorization on a Matrix A.

    Parameters:
        A: random generated or selected matrix A which is square (for now)
        n: dimension of the matrix
        routine: specifies which routine to use within LU Factorization:
                 1 for no pivoting
                 2 for partial row pivoting

    Returns:
        Either just Matrix A decomposed into L and U stored in a single 2D array or
        returns Matrix A and Permutation vector P which stores the indexes of the permutations performed
        during LU Factorization with partial row pivoting. i.e.
        Routine 1 Returns:
            A: Matrix A comprised of L and U stored in 2D array (A) with 0's and 1's not stored
        Routine 2 Returns:
            A: Matrix A comprised of L and U stored in 2D array (A) with 0's and 1's not stored
            P: Permutation Vector storing the indices of the permutations performed during LU Factorization
    """

    if routine == 1:
        for i in range(n):
            alpha = A[i][i] # Setting the alpha i,i in computing lambda

            # Error Tolerance Threshold for Alpha
            if abs(alpha) < 1e-12:
                print(f"Error: Near-zero pivot encountered at row {i}.")
                return None

            # # Checking if the alpha i,i element is zero
            # if alpha == 0:
            #     print("Warning: there is a zero in the diagonal and the LU will no longer proceed.")
            #     return None

            # Loop over rows below current row i
            for k in range(i + 1, n):
                lamb = A[k][i]/alpha # Computes lambda
                A[k][i] = lamb # Storing the lambdas

                # Update the remaining elements in the kth row
                for j in range(i + 1, n):
                    A[k][j] = A[k][j] - lamb * A[i][j] # Performing the reduction

        return A

    elif routine == 2:
        # Initialize P with i + 1 for readability on output, must remember for permutating back
        P = [i for i in range(n)]
        for i in range(n):
            # Find row with the largest absolute value in column i
            max_row = i # initialize max row to the current row i
            max_value = abs(A[i][i]) # initialize the max row with the corresponding absolute value

            # Iterating rows below to find row with max absolute value
            for k in range(i + 1, n):
                if abs(A[k][i]) > max_value:
                    max_value = abs(A[k][i])
                    max_row = k

            # Check if zero pivots
            if A[max_row][i] == 0:
                print("Warning: Zero pivot found, LU cannot proceed.")
                return None, P

            # Swap Rows if maximal element found in column i and does the same for Permutation Vector
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                P[i], P[max_row] = P[max_row], P[i]
                # Print which rows were swapped
                print(f"Row {i} swapped with row {max_row} for partial pivoting to proceed.")


            # Continue LU Factorization after pivoting
            alpha = A[i][i]
            if abs(alpha) < 1e-12:
                print(f"Error: Near-zero pivot encountered at row {i}.")
                return None, P
            for k in range(i + 1, n):
                lamb = A[k][i]/alpha
                A[k][i] = lamb

                for j in range(i + 1, n):
                    A[k][j] = A[k][j] - lamb * A[i][j]

        return A, P

    else:
        print(f"Error: Invalid routine number chosen: {routine}. Please enter 1 for no pivoting or 2 for partial pivoting.")
        return None

def row_permutation(P, A, n):
    A_perm = [[0] * n for i in range(n)]

    for i in range(n):
        A_perm[i] = A[P[i]]

    return A_perm

def rand_row_perm(A, n):
    P_rand = [i for i in range(n)]
    random.shuffle(P_rand)

    A_perm = [[0] * n for i in range(n)]

    for i in range(n):
        A_perm[i] = A[P_rand[i]]

    return A_perm

def mat_copy(A, n):
    A_copy = [[0] for i in range(n)]
    for i in range(n):
        for k in range(n):
            A_copy[i][k] = A[i][k]

    return A_copy


# Outputting Specified Routine
# a = int(input("Lower Bound for Random Number in Vector: "))
# b = int(input("Upper Bound for Random Number in Vector: "))
# routine = int(input("Routine Number: "))
# n = int(input("Dimension of Matrix: "))
# U = my_package.generate_nonsingular_upper_triangular(a, b, n)
# L = my_package.generate_unit_lower_triangular(a, b, n)
# LU = my_package.combine_upper_lower(L, U, n)
# A = my_package.UL_mult_COMB(LU, n)
# A_copy = [[0] * n for i in range(n)]
# for i in range(n):
#     for k in range(n):
#         A_copy[i][k] = A[i][k]
# #
# # # print("Matrix U is: ")
# # # for r in U:
# # #     print(r)
# # #
# print("Matrix A is: ")
# for r in A:
#     print(r)
# #
# print("Matrix A_copy is: ")
# for r in A_copy:
#     print(r)
#
# # if routine == 1:
# #     B = LU_factor(A, n, routine)
# #     if B is not None:
# #         print("Resulting Matrix B:")
# #         for r in B:
# #             print(r)
# #     else:
# #         print("LU Factorization has failed.")
# # elif routine == 2:
# #     B, P = LU_factor(A, n, routine)
# #     if B is not None:
# #         print("Resulting Matrix B:")
# #         for r in B:
# #             print(r)
# #
# #         print("\nPermutation Vector P is:")
# #         print(P)
# #
# #         A_perm = row_permutation(P, A, n)
# #         print("\nPermutation of A is: ")
# #         for r in A_perm:
# #             print(r)
# #
# #         # To verify the correctness of the factorization, calculate A_tilda
# #         L, U = B, B  # Assuming you have L and U from your combined matrix
# #         A_tilda = [[0] * n for _ in range(n)]
# #
# #         for i in range(n):
# #             for j in range(n):
# #                 for k in range(n):
# #                     A_tilda[i][j] += L[i][k] * U[k][j]
# #
# #         print("\nMatrix A_tilda is: ")
# #         for r in A_tilda:
# #             print(r)
# #
# #         # Check the difference
# #         diff_A = [[A_perm[i][j] - A_tilda[i][j] for j in range(n)] for i in range(n)]
# #         print("\nDifference of Matrix A is: ")
# #         for r in diff_A:
# #             print(r)
# #
# #     else:
# #         print("LU Factorization has failed due to a zero on the diagonal.")
# # else:
# #     B = LU_factor(A, n, routine)
#
#
# Step 1: Perform LU Factorization for partial pivoting
# B, P = LU_factor(A, n, routine)
# print("Matrix B is: ")
# for r in A:
#     print(r)
#
# print("Permutation Vector P is: ", P)
#
#
# # Step 2: Permute A
# A_perm = row_permutation(P, A_copy, n)
# print("Matrix A_perm is: ")
# for r in A_perm:
#     print(r)
#
# # Step 3: Reconstruct A_tilde from B
# A_reconstruct = my_package.UL_mult_COMB(B, n)
# print("Matrix A_reconstruct is: ")
# for r in A_reconstruct:
#     print(r)
#
# # Step 4: Check Differences
# difference = [[0] * n for i in range(n)]
# for i in range(n):
#     for k in range(n):
#         difference[i][k] = A_reconstruct[i][k] - A_perm[i][k]
# print("Difference of Matrix A:")
# for row in difference:
#     print(row)
#


'''TESTER'''

'''User inputs to determine dimensions and Bounds for Random Numbers and specify which routine'''
# dim_min = int(input("Minimum dimension to be tested: "))
# dim_max = int(input("Maximum dimension to be tested: "))
# num_tests = int(input("Number of tests per dimension: "))
# a = int(input("Lower Bound for Random Number: "))
# c = int(input("Upper Bound for Random Number: "))
routine = int(input("Routine Number (1 for no pivoting, 2 for partial pivoting): "))
# #n = int(input("Number of dimensions: "))
# ratio = float(input("Ratio factor (float) for scaling off-diagonal elements of U (default value is 1.0): "))

def LU_factor_go(dim_min, dim_max, num_tests, a, c, routine, ratio):
    """
    Description:
    Performs LU Factorization on a Matrix A.
    Parameters:
        dim_min: minimum dimension of matrices and vectors
        dim_max: maximum dimension of matrices and vectors
        num_tests: number of tests per dimension
        a: lower bound on random numbers/floating numbers
        c: upper bound on random numbers/floating numbers
        routine: specifies which routine to use within LU Factorization:
            1 for no pivoting
            2 for partial row pivoting
        ratio: specifies ratio to be used to scale off-diagonal elements of U
    Returns:
    """
    # Check Valid Routine before continuing
    if routine not in [1, 2]:
        print("Invalid routine number chosen. Please choose 1 for no pivoting or 2 for partial pivoting.")
        return

    # No Pivoting Routine
    if routine == 1:
        dimensions = list(range(dim_min, dim_max + 1))
        fact_acc = []
        x_error = []
        resid_error = []
        growth_factor = []

        for n in dimensions:
            fact_acc_for_n = []
            x_error_for_n = []
            resid_error_for_n = []
            growth_factor_for_n = []

            for i in range(num_tests):
                # Inputs for tester
                U = my_package.generate_nonsingular_upper_triangular(a, c, n, ratio)
                #L = my_package.generate_unit_lower_triangular(a, c, n)
                L = my_package.generate_cond_unit_lower_triangular(a, n)
                x = my_package.generate_float_1D_vector(a, c, n)
                LU = my_package.combine_upper_lower(L, U, n)
                A = my_package.UL_mult_COMB(LU, n)
                #A_rand = rand_row_perm(A, P, n)
                A_copy = [[0] * n for j in range(n)]
                for j in range(n):
                    for k in range(n):
                        A_copy[j][k] = A[j][k]

                # Custom Algorithm for LU Factorization with no pivoting in this routine of tester
                A_factor = LU_factor(A, n, routine)
                A_factor_abs_np = np.abs(A_factor)
                A_factor_abs = A_factor_abs_np.tolist()

                # Handles case when threshold is hit and continues testing beyond given case
                if A_factor is None:
                    print(f"Test case {i+1}/{num_tests} skipped due to threshold hit in pivot.")
                    continue

                # Recompute A (C) from A_factor
                C = my_package.UL_mult_COMB(A_factor, n)

                # Calculate Factorization Accuracy
                diff_mat = [[0] * n for j in range(n)]
                for j in range(n):
                    for k in range(n):
                        diff_mat[j][k] = A_copy[j][k] - C[j][k]

                diff_mat_norm = my_package.frob_norm(diff_mat)
                A_norm = my_package.frob_norm(A_copy)

                err_diff_A = diff_mat_norm/A_norm if A_norm != 0 else float('inf')

                ## Appending accuracy to list
                fact_acc_for_n.append(np.log(err_diff_A + 1e-12))
                #fact_acc_for_n.append(err_diff_A)

                # Compute Ax = b to find b
                # Converting to np array for dot product and convert back to list
                A_np = np.array(A_copy)
                x_np = np.array(x)
                b_np = np.dot(A_np, x_np)
                b = b_np.tolist()

                # Solve Ly = b
                y = solve_Lb(A_factor, b, n)

                # Solve Ux = y
                x_comp = solve_Ux(A_factor, y, n)

                # Compute Error for x
                diff_x = [0] * n
                for j in range(n):
                    diff_x[j] = x[j] - x_comp[j]

                diff_x_norm = my_package.vec_2_norm(diff_x)
                x_norm = my_package.vec_2_norm(x)

                err_diff_x = diff_x_norm/x_norm if x_norm != 0 else float('inf')

                ## Appending Error to list
                x_error_for_n.append(np.log(err_diff_x + 1e-12))
                #x_error_for_n.append(err_diff_x)

                # Compute Residual
                x_comp_np = np.array(x_comp)

                b_comp_np = np.dot(A_np, x_comp_np)
                b_comp = b_comp_np.tolist()


                diff_b = [0] * n
                for j in range(n):
                    diff_b[j] = b[j] - b_comp[j]

                diff_b_norm = my_package.vec_2_norm(diff_b)
                b_norm = my_package.vec_2_norm(b)

                residual_diff_b = diff_b_norm/b_norm if b_norm != 0 else float('inf')

                ## Appending to error to list
                resid_error_for_n.append(np.log(residual_diff_b + 1e-12))
                #resid_error_for_n.append(residual_diff_b)

                # Compute Growth Factor
                A_factor_grow = my_package.UL_mult_COMB(A_factor_abs, n)

                A_factor_grow_norm = my_package.frob_norm(A_factor_grow)
                growth = A_factor_grow_norm / A_norm

                ## Appending Growth Factor to list
                growth_factor_for_n.append(growth)


            # Averages
            fact_acc.append(np.mean(fact_acc_for_n))
            x_error.append(np.mean(x_error_for_n))
            resid_error.append(np.mean(resid_error_for_n))
            growth_factor.append(np.mean(growth_factor_for_n))

    # Partial Pivoting Routine
    if routine == 2:
        dimensions = list(range(dim_min, dim_max + 1))
        fact_acc = []
        x_error = []
        resid_error = []
        growth_factor = []

        for n in dimensions:
            fact_acc_for_n = []
            x_error_for_n = []
            resid_error_for_n = []
            growth_factor_for_n = []

            for i in range(num_tests):
                # Inputs for tester
                U = my_package.generate_nonsingular_upper_triangular(a, c, n, ratio)
                L = my_package.generate_cond_unit_lower_triangular(a, n)
                x = my_package.generate_float_1D_vector(a, c, n)
                LU = my_package.combine_upper_lower(L, U, n)
                A = my_package.UL_mult_COMB(LU, n)
                A_rand = rand_row_perm(A, n)
                A_copy = copy.deepcopy(A_rand)

                # Perform LU Factorization
                B, P = LU_factor(A_rand, n, routine)
                print("Permutation Vector P is: ", P)
                B_abs_np = np.abs(B)
                B_abs = B_abs_np.tolist()

                # Handles case when threshold is hit and continues testing beyond given case
                if B is None:
                    print(f"Test case {i+1}/{num_tests} skipped due to threshold hit in pivot.")
                    continue

                U_np = np.array(U)
                condition_number = np.linalg.cond(U_np)
                #print("Matrix U_np is: ", U_np)
                print("Condition number of U is: ", condition_number)

                # Step 2: Permute A
                A_perm = row_permutation(P, A_copy, n)

                # Recompute A (C) from A_factor
                C = my_package.UL_mult_COMB(B, n)

                # Step 4: Check Differences
                diff_mat_pivot = [[0] * n for i in range(n)]
                for i in range(n):
                    for k in range(n):
                        diff_mat_pivot[i][k] = C[i][k] - A_perm[i][k]

                # Factorization Accuracy
                diff_mat_pivot_norm = my_package.frob_norm(diff_mat_pivot)
                A_norm = my_package.frob_norm(A_perm)
                err_diff_A_perm = diff_mat_pivot_norm / A_norm if A_norm != 0 else float('inf')
                #fact_acc_for_n.append(np.log(err_diff_A_perm + 1e-10))
                fact_acc_for_n.append(err_diff_A_perm)

                # Compute Ax = b to find b
                # We have LUx = b where LU now equals C
                A_copy_np = np.array(A_copy)
                x_np = np.array(x)
                b_np = np.dot(A_copy_np, x_np)
                b = b_np.tolist()
                Pb = row_permutation(P, b, n)

                # Solve Ly = b
                y = solve_Lb(B, Pb, n)

                # Solve Ux = y
                x_comp = solve_Ux(B, y, n)

                # Compute Error for x
                diff_x = [0] * n
                for j in range(n):
                    diff_x[j] = x[j] - x_comp[j]

                diff_x_norm = my_package.vec_2_norm(diff_x)
                x_norm = my_package.vec_2_norm(x)

                err_diff_x = diff_x_norm / x_norm if x_norm != 0 else float('inf')
                x_error_for_n.append(err_diff_x)
                #x_error_for_n.append(np.log(err_diff_x + 1e-10))

                # Compute Residual and Residual Error
                x_comp_np = np.array(x_comp)
                A_perm_np = np.array(A_perm)

                b_comp_np = np.dot(A_perm_np, x_comp_np)
                b_comp = b_comp_np.tolist()

                diff_b = [0] * n
                for j in range(n):
                    diff_b[j] = Pb[j] - b_comp[j]

                diff_Pb_norm = my_package.vec_2_norm(diff_b)
                Pb_norm = my_package.vec_2_norm(Pb)

                residual_diff_Pb = diff_Pb_norm / Pb_norm if Pb_norm != 0 else float('inf')

                #resid_error_for_n.append(np.log(residual_diff_Pb + 1e-10))
                resid_error_for_n.append(residual_diff_Pb)

                # Compute Growth Factor
                B_grow = my_package.UL_mult_COMB(B_abs, n)

                B_grow_norm = my_package.frob_norm(B_grow)
                growth = B_grow_norm / A_norm

                growth_factor_for_n.append(growth)

            # Averages
            fact_acc.append(np.mean(fact_acc_for_n))
            x_error.append(np.mean(x_error_for_n))
            resid_error.append(np.mean(resid_error_for_n))
            growth_factor.append(np.mean(growth_factor_for_n))



    # Plots of Errors and Accuracy
    plt.figure(figsize=(24, 6))

    # Factorization Accuracy Subplot
    plt.subplot(1, 4, 1)
    plt.plot(dimensions, fact_acc, label='Log Factorization Error', marker='o', color='blue', linestyle='-')
    plt.xlabel('Matrix Dimension (n)', fontsize=14)
    plt.ylabel('Log of Frobenius Norm Difference', fontsize=14)
    plt.title('Log Factorization Error', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)  # Reference line at y=0
    plt.grid(True)
    plt.legend(fontsize=12)

    # Solver Error Subplot
    plt.subplot(1, 4, 2)
    plt.plot(dimensions, x_error, label='Log Vector x Error', marker='s', color='orange', linestyle='-')
    plt.xlabel('Matrix Dimension (n)', fontsize=14)
    plt.ylabel('Log of Vector 2 Norm Difference', fontsize=14)
    plt.title('Log Vector x Error', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)  # Reference line at y=0
    plt.grid(True)
    plt.legend(fontsize=12)

    # Residual Error Subplot
    plt.subplot(1, 4, 3)
    plt.plot(dimensions, resid_error, label='Log Residual Error', marker='^', color='green', linestyle='-')
    plt.xlabel('Matrix Dimension (n)', fontsize=14)
    plt.ylabel('Log of Residual Norm Difference', fontsize=14)
    plt.title('Log Residual Error', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)  # Reference line at y=0
    plt.grid(True)
    plt.legend(fontsize=12)

    # Growth Factor Subplot
    plt.subplot(1, 4, 4)
    plt.plot(dimensions, growth_factor, label='Growth Factor', marker='o', color='blue', linestyle='-')
    plt.xlabel('Matrix Dimension (n)', fontsize=14)
    plt.ylabel('Growth Factor', fontsize=14)
    plt.title('Growth Factor', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)
    plt.grid(True)
    plt.legend(fontsize=12)

    plt.suptitle(f'Plots for Ratio Factor of {ratio}', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjusts subplots to fit into the figure area.
    plt.show()


#LU_factor_go(dim_min, dim_max, num_tests, a, c, routine, ratio)


"""Empirical Cases and Tests"""

# Empirical Test 1
print("\nWe first look at Empirical Test 1 for A and A_1:")
A_t1_n = [[1, 0, 0, 0, 0],
          [0, 2, 0, 0, 0],
          [0, 0, 3, 0, 0],
          [0, 0, 0, 4, 0],
          [0, 0, 0, 0, 5]]
A_t1_p = [[1, 0, 0, 0, 0],
          [0, 2, 0, 0, 0],
          [0, 0, 3, 0, 0],
          [0, 0, 0, 4, 0],
          [0, 0, 0, 0, 5]]

A_1_t1_n = [[5, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 1]]
A_1_t1_p = [[5, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 1]]

n = len(A_t1_n)

print("\nMatrix A_t1 is: ")
for r in A_t1_n:
    print(r)

# No Pivoting for First A
B_t1_r1 = LU_factor(A_t1_n, n, routine=1)
print("\nMatrix B_t1 from A_t1 with no pivoting is: ")
for r in B_t1_r1:
    print(r)

# Partial Pivoting for First A
B_t1_r2, P_t1 = LU_factor(A_t1_p, n, routine=2)
print("\nMatrix B_t1 from A_t1 with partial pivoting is: ")
for r in B_t1_r2:
    print(r)
print("\nPermutation Vector P_t1 is: ", P_t1)


print("\nMatrix A_1_t1 is: ")
for r in A_1_t1_n:
    print(r)

# No Pivoting for Second A
B_1_t1_r1 = LU_factor(A_1_t1_n, n, routine=1)
print("\nMatrix B_1_t1 from A_1_t1 with no pivoting is: ")
for r in B_1_t1_r1:
    print(r)

# Partial Pivoting for Second A
B_1_t1_r2, P_1_t1 = LU_factor(A_1_t1_p, n, routine=2)
print("\nMatrix B_1_t1 from A_1_t1 with partial pivoting is: ")
for r in B_1_t1_r2:
    print(r)
print("\nPermutation Vector P_1_t1 is: ", P_1_t1)

print("\nThis concludes Empirical Test 1 for A_t1 and A_1_t1.\n")


# Empirical Test 2
print("\nWe now look at Empirical Test 2:")
A_t2_n = [[0, 0, 0, 0, 1],
          [0, 0, 0, 2, 0],
          [0, 0, 3, 0, 0],
          [0, 4, 0, 0, 0],
          [5, 0, 0, 0, 0]]
A_t2_p = [[0, 0, 0, 0, 1],
          [0, 0, 0, 2, 0],
          [0, 0, 3, 0, 0],
          [0, 4, 0, 0, 0],
          [5, 0, 0, 0, 0]]

A_1_t2_n = [[0, 0, 0, 0, 5],
            [0, 0, 0, 4, 0],
            [0, 0, 3, 0, 0],
            [0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0]]
A_1_t2_p = [[0, 0, 0, 0, 5],
            [0, 0, 0, 4, 0],
            [0, 0, 3, 0, 0],
            [0, 2, 0, 0, 0],
            [1, 0, 0, 0, 0]]

n = len(A_t2_n)

print("\nMatrix A_t2 is: ")
for r in A_t2_n:
    print(r)

# B_t2_r1 = LU_factor(A_t2, n, routine=1)
# print("\nMatrix B_t2 with no pivoting is: ")
# for r in B_t2_r1:
#     print(r)
B_t2_r2, P_t2_r2 = LU_factor(A_t2_p, n, routine=2)
print("\nMatrix B_t2 with partial pivoting is: ")
for r in B_t2_r2:
    print(r)
print("\nPermutation Vector P_t2 is: ", P_t2_r2)

print("\nMatrix A_1_t2 is: ")
for r in A_1_t2_n:
    print(r)

# B_1_t2_r1 = LU_factor(A_1_t2, n, routine=1)
# print("\nMatrix B_1_t2 with no pivoting is: ")
# for r in B_1_t2_r1:
#     print(r)
B_1_t2_r2, P_1_t2_r2 = LU_factor(A_1_t2_p, n, routine=2)
print("\nMatrix B_1_t2 with partial pivoting is: ")
for r in B_1_t2_r2:
    print(r)
print("\nPermutation Vector P_1_t2 is: ", P_1_t2_r2)

print("\nThis concludes Empirical Test 2 for A_t2 and A_1_t2.\n")

# Empirical Test 3


print("\nThis concludes Empirical Test 3 for A_t3.\n")


# Empirical Test 4


print("\nThis concludes Empirical Test 4 for A_t4.\n")


# Empirical Test 5
print("\nWe now look at Empirical Test 5: ")
A_t5_n = [[2, 0, 0, 0, 0],
          [3, 2, 0, 0, 0],
          [4, 3, 2, 0, 0],
          [5, 4, 3, 2, 0],
          [6, 5, 4, 3, 2]]
A_t5_p = [[2, 0, 0, 0, 0],
          [3, 2, 0, 0, 0],
          [4, 3, 2, 0, 0],
          [5, 4, 3, 2, 0],
          [6, 5, 4, 3, 2]]

n = len(A_t5_n)

print("\nMatrix A_t5 is: ")
for r in A_t5_n:
    print(r)

# No pivoting
B_t5_r1 = LU_factor(A_t5_n, n, routine=1)
print("\nMatrix B_t5 with no pivoting is: ")
for r in B_t5_r1:
    print(r)

# Partial Pivoting
B_t5_r2, P_t5_r2 = LU_factor(A_t5_p, n, routine=2)
print("\nMatrix B_t5 with partial pivoting is: ")
for r in B_t5_r2:
    print(r)
print("\nPermutation Vector P is: ", P_t5_r2)

print("\nThis concludes Empirical Test 1 for A_t5.\n")

# Empirical Test 6

# Empirical Test 7
print("\nWe now look at Empirical Test 7:")
A_t7_n = [[1, 0, 0, 0, 1],
          [-1, 1, 0, 0, 1],
          [-1, -1, 1, 0, 1],
          [-1, -1, -1, 1, 1],
          [-1, -1, -1, -1, 1]]

A_t7_p = [[1, 0, 0, 0, 1],
     [-1, 1, 0, 0, 1],
     [-1, -1, 1, 0, 1],
     [-1, -1, -1, 1, 1],
     [-1, -1, -1, -1, 1]]

n = len(A_t7_n)

print("\nMatrix A_t7 is: ")
for r in A_t7_n:
    print(r)

# No Pivoting
B_t7_r1 = LU_factor(A_t7_n, n, routine=1)
print("\nMatrix B_t7 with no pivoting is: ")
for r in B_t7_r1:
    print(r)

# Partial Pivoting
B_t7_r2, P_t7_r2 = LU_factor(A_t7_p, n, routine=2)
print("\nMatrix B_t7 with partial pivoting is: ")
for r in B_t7_r2:
    print(r)
print("\nPermutation Vector P is: ", P_t7_r2)

print("\nThis concludes Empirical Test 7 for A_t7.\n")
# Empirical Test 8




