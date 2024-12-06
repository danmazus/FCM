import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# f(x) function for standard functions
# def f(x):
#     f = pow(x, 3) + (-3*x) + 1
#     return f

# rho function for higher order single root functions
def p(rho, d):
    return lambda x: (x - rho) ** d

def p_2(rho):
    return lambda x: x**3 - ((rho ** 2) * x)

def p_3(rho_1, rho_2):
    return lambda x: x * (x - rho_1) * (x - rho_2)

def dp(rho, d):
    return lambda x: d * (x - rho) ** (d-1)

def dp_2(rho):
    alpha = rho / (np.sqrt(3))
    return lambda x: 3 * (x - alpha) * (x + alpha)

def display_panda_df(log):
    df = pd.DataFrame(log, columns=["Iteration", "x_k", "f(x_k)", "Ratio"])

    pd.set_option("display.float_format", "{:.7f}".format,
                  "display.max_rows", 20)
    print(df)

# Regula-Falsi
def reg_fal_method(f, a0, b0, max_iter, tol = 1e-6):
    if f(a0) * f(b0) >= 0:
        raise ValueError("The function values at a0 and b0 must have opposite signs")

    # Setting x_0 as either a_0 or b_0
    # if abs(f(a0)) < abs(f(b0)):
    #     x = a0
    # else:
    #     x = b0
    x = a0
    #x = b0

    # Setting initial values for iterations
    a_k = a0
    b_k = b0

    # Compute Initial q_0
    q_k = (f(b_k) - f(a_k)) / (b_k - a_k)

    # Iteration start
    k = 0

    results = []

    # Finding root
    while k < max_iter:

        # Compute x_(k+1)
        x_next = x - (f(x) / q_k)

        interval_size = abs(b_k - a_k)

        # Check for convergence
        if abs(f(x_next)) < tol or interval_size < tol:
            return x_next, k + 1

        # Setting Next values depending on what the signs are
        if f(x_next) * f(a_k) < 0:
            a_k_next = a_k
            b_k_next = x_next
        elif f(x_next) * f(b_k) < 0:
            a_k_next = x_next
            b_k_next = b_k
        else:
            # This is a return statement as we have found the root directly
            return x_next, k + 1

        # Compute q_(k+1)
        q_next = (f(b_k_next) - f(a_k_next)) / (b_k_next - a_k_next)

        # Update Values for next iterations
        a_k = a_k_next
        b_k = b_k_next
        x = x_next
        q_k = q_next
        k += 1

    raise ValueError("Regula Falsi method did not converge within maximum number of iterations")



# Secant Method
def secant_method(f, x0, x1, rho, max_iter, tol = 1e-6):

    k = 0
    x_k0 = x0    # x_(-1)
    x_k1 = x1    # x_0
    x_true = rho
    results = []

    #if f(x_k0) * f(x_k1) >= 0:
     #   raise ValueError("The function values at x0 and x1 must have opposite signs")

    while k < max_iter:
        # Compute q_k
        q_k = (f(x_k1) - f(x_k0)) / (x_k1 - x_k0)

        # Compute x_(k+1)
        x_k_next = x_k1 - f(x_k1) / q_k

        err_k = abs(x_k1 - x_true)
        err_k_next = abs(x_k_next - x_true)

        ratio = abs(err_k_next) / abs(err_k)

        results.append((k, x_k1, err_k, ratio))

        # Check for direct convergence
        if abs(x_k_next - x_k1) < tol:
            results.append((k+1, x_k_next, err_k_next, ratio))
            return x_k_next, k + 1, results

        #abs(f(x_k_next)) < tol or

        # Update values for next iteration
        x_k0 = x_k1
        x_k1 = x_k_next
        k += 1

    raise ValueError("Secant method did not converge within the maximum number of iterations")



# Newton's Method
def newton_method(f, df, x0, rho, max_iter, m = 1.0, tol = 1e-6):
    # Set initial values
    global err_k_next, ratio
    k = 0
    x = x0
    x_true = rho
    results = []

    while k < max_iter:
        # Compute q_k = f'(x_k)
        f_val = f(x)
        q_k = df(x)

        # if abs(q_k) < tol:
        #     if abs(f_val) < tol:
        #         results.append((k, x, f_val, ratio))
        #         return x, k + 1, results
        #     else:
        #         raise ValueError(f"q_k = {q_k} is close to zero but f(x) = {f_val} is not close to zero")

        # Comptue x_(k+1)
        x_next = x - (m * (f_val / q_k))

        err_k = abs(x - x_true)
        err_k_next = abs((x_next - x_true))

        ratio = abs(err_k_next) / abs(err_k)


        results.append((k, x, f_val, ratio))



        if abs(f(x_next)) < tol or abs(x_next - x) < tol:
            results.append((k+1, x_next, f(x_next), ratio))
            return x_next, k + 1, results

        x = x_next
        k += 1

    raise ValueError(f"Newton's method did not converge within {max_iter}. Last x_k = {x}, iteration = {k}")



# Steffenson's Method
def steff_method(f, x0, rho, max_iter, tol = 1e-6):
    x = x0
    x_true = rho
    k = 0
    results = []
    while k < max_iter:
        # First way of implementing steffenson's method
        theta = (f(x + f(x)) - f(x)) / f(x)
        g = x - (f(x) / theta)
        x_next = g

        err_k = abs(x - x_true)
        err_k_next = abs(x_next - x_true)

        ratio = err_k_next / (err_k)

        results.append((k, x, err_k, ratio))

        # Convergence check for both direct or x_k+1 - x_k convergence
        if abs(f(x_next)) < tol or abs(x_next - x) < tol:
            results.append((k+1, x_next, err_k_next, ratio))
            return x_next, k + 1, results

        x = x_next
        k += 1

    raise ValueError("Steffenson's method did not converge within the maximum number of iterations")


"""Higher Order Roots Questions"""
# Assigning Values
rho = 1.9
d = [2, 3, 4, 5, 6, 7, 8, 9, 10]
m_minus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
m_plus = [3, 4, 5, 6, 7, 8, 9, 10, 11]
x0 = [1, 1.5, 2, 2.5]
x1 = [1.5, 1.7, 2.1, 2.3]
a0 = [0, 0.5, 1, 1.5]

# Problem 1
print("\n-------------------------------")
print("PROBLEM 1")
print("-------------------------------")
print(f"\nUsing Standard Newton's Method:")
# fig, ax1 = plt.subplots(figsize=(10,6))
# ax1.set_xlabel("Multiplicity of Root")
# ax1.set_ylabel("Iterations")
# ax1.set_title("Root for Standard Newton's Method")
# ax1.grid(True)
for i in d:
    for k in x0:
        f = p(rho, i)
        df = dp(rho, i)

        try:
            solution, iteration, log = newton_method(f, df, k, rho, max_iter=1000, m=1.0, tol=1e-6)
            print(f"\nd = {i}, x0 = {k}, Root = {solution:.7f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {k}, Error: {e}")

# Problem 2
print("\n-------------------------------")
print("PROBLEM 2")
print("-------------------------------")
print(f"\nUsing Newton's Method when d = m")
for i in d:
    for k in x0:
        f = p(rho, i)
        df = dp(rho, i)

        try:
            solution, iteration, log = newton_method(f, df, k, rho, max_iter=1000, m=i, tol=1e-6)
            print(f"\nd = {i}, x0 = {k}, Root = {solution:.7f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {k}, Error: {e}")


# # Problem 3
print("\n-------------------------------")
print("PROBLEM 3")
print("-------------------------------")
print(f"\nUsing Steffenson's Method")
# # fig, ax3 = plt.subplots(figsize=(10,6))
# # ax3.set_xlabel("Multiplicity of Root")
# # ax3.set_ylabel("Iterations")
# # ax3.set_title("Root for Steffenson's Method")
# # ax3.grid(True)
# # colors = ['red', 'green', 'blue', 'purple', 'black', 'yellow', 'brown']
# # markers = ['o', '^', 'v', 'x', 's', 'd', 'h']
#
for i in d:
    f = p(rho, i)

    for k in x0:

        try:
            solution, iteration, log = steff_method(f, k, rho, max_iter=1000, tol=1e-6)
            print(f"\nd = {i}, x0 = {k}, Root = {solution:.7f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {k}, Error: {e}")
#
# #
# # # for i, k in enumerate(x0):
# # #     ax3.plot(d, iterations[k], label=f"x0 = {k}", marker = markers[i],color=colors[i])
# # #
# # # plt.legend(title="Initial Guess x0", loc="best")
# # # plt.show()
# #

# # Problem 4
print("\n-------------------------------")
print("PROBLEM 4")
print("-------------------------------")
print(f"\nUsing Regula Falsi Method")
for i in d:
    for k in a0:
        f = p(rho, i)

        try:
            solution, iteration = reg_fal_method(f, k, 2.1, max_iter=20000, tol=1e-6)
            print(f"d = {i}, a0 = {k}, b0 = 2.1, Root = {solution:.6f}, Iterations = {iteration}")

        except ValueError as e:
            print(f"d = {i}, a0 = {k}, b0 = 2.1, Error: {e}")

print(f"\nUsing Secant Method")
for i in d:
    f = p(rho, i)

    for k in range(len(x0)):
        initial_x0 = x0[k]
        initial_x1 = x1[k]

        try:
            solution, iteration, log = secant_method(f, initial_x0, initial_x1, rho, max_iter=1000, tol=1e-6)
            print(f"\nd = {i}, x0 = {initial_x0}, x1 = {initial_x1}, Root = {solution:.7f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {initial_x0}, x1 = {initial_x1}, Error: {e}")

# Problem 5
print("\n-------------------------------")
print("PROBLEM 5")
print("-------------------------------")
print(f"\nUsing Each Method when d = 1")
for k in x0:
    f = p(rho, 1)
    df = dp(rho, 1)

    try:
        print(f"\nUsing Standard Newton's Method for x0 = {k}:")
        solution, iteration, log = newton_method(f, df, k, rho, max_iter=1000, m=1.0, tol=1e-6)
        print(f"d = 1, m = 1, x0 = {k}, Root = {solution:.6f}, Iterations = {iteration}")
        display_panda_df(log)

    except ValueError as e:
        print(f"d = 1, x0 = {k}, Error: {e}")

    try:
        print(f"\nUsing Steffenson's Method for x0 = {k}:")
        solution, iteration, log = steff_method(f, k, rho, max_iter=1000, tol=1e-6)
        print(f"d = 1, x0 = {k}, Root = {solution:.6f}, Iterations = {iteration}")

    except ValueError as e:
        print(f"d = 1, x0 = {k}, Error: {e}")

for k in range(len(x0)):
    f = p(rho, 1)
    initial_x0 = x0[k]
    initial_x1 = x1[k]

    try:
        print(f"\nUsing Secant Method for x0 = {initial_x0} and x1 = {initial_x1}:")
        solution, iteration, log = secant_method(f, initial_x0, initial_x1, rho, max_iter=1000, tol=1e-6)
        print(f"d = 1, x0 = {initial_x0}, x1 = {initial_x1}, Root = {solution:.6f}, Iterations = {iteration}")
        display_panda_df(log)

    except ValueError as e:
        print(f"d = 1, x0 = {initial_x0}, x1 = {initial_x1}, Error: {e}")


# Problem 6
print("\n-------------------------------")
print("PROBLEM 6")
print("-------------------------------")
print(f"\nUsing Newton's Method for m > d")
for i in d:
    for k in x0:
        f = p(rho, i)
        df = dp(rho, i)
        try:
            solution, iteration, log = newton_method(f, df, k, rho, max_iter=1000, m=i+1, tol=1e-6)
            print(f"\nd = {i}, m = {i+1}, x0 = {k}, Root = {solution:.6f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {k}, Error: {e}")

print(f"\nUsing Newton's Method for m < d")
for i in d:
    for k in x0:
        f = p(rho, i)
        df = dp(rho, i)
        back = i - 1
        try:
            solution, iteration, log = newton_method(f, df, k, rho, max_iter=1000, m=back, tol=1e-6)
            print(f"\nd = {i}, m = {i-1}, x0 = {k}, Root = {solution:.6f}, Iterations = {iteration}")
            display_panda_df(log)

        except ValueError as e:
            print(f"d = {i}, x0 = {k}, Error: {e}")