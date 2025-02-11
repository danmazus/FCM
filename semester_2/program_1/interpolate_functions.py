import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # f(x)=4+3x+2x^2+x^3
    return 4 + 3 * x + 2 * np.power(x,2) + np.power(x,3)

def f_2(x):
    return abs(x) + 0.5 * x - x ** 2

def chebyshev_1(n):
    x_mesh = np.zeros(n+1)

    for i in range(n+1):
        x_mesh[i] = np.cos(((2*i + 1) * np.pi) / (2 * n + 2))

    return x_mesh

def chebyshev_2(n):
    x_mesh = np.zeros(n+1)

    for i in range(n+1):
        x_mesh[i] = np.cos((i * np.pi)/ n)

    return x_mesh

def x_mesh_order(x_mesh, flag):
    n = len(x_mesh)
    x_mesh_order = x_mesh.copy()

    # Decreasing order of x-values in mesh
    if flag == 1:
        for i in range(n):
            for j in range(0, n-i-1):
                if x_mesh_order[j] < x_mesh_order[j+1]:
                    x_mesh_order[j], x_mesh_order[j+1] = x_mesh_order[j+1], x_mesh_order[j]

    # Increasing order of x-values in mesh
    elif flag == 2:
        for i in range(n):
            for j in range(0, n-i-1):
                if x_mesh_order[j] > x_mesh_order[j+1]:
                    x_mesh_order[j], x_mesh_order[j+1] = x_mesh_order[j+1], x_mesh_order[j]

    elif flag == 3:
        # Picking x_0 as the max by getting the index of the max through argmax and then reordering terms
        x_max_index = np.argmax(np.abs(x_mesh_order))
        x_mesh_order[0], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[0]

        # Rest of the values in the mesh
        for i in range(1, n):
            # Initializing product vector
            product = np.ones(n)

            # Leja ordering product
            for j in range(i, n):
                product[j] = product[j-1] * (x_mesh_order[j] - x_mesh_order[i-1])

            # Indexing through to find the next point with max product and add i to get correct index
            # Adding i is needed as this will adjust the index in the subarray to the original array location
            x_max_index = np.argmax(np.abs(product[i:])) + i

            # Reordering mesh based on the point with max product
            x_mesh_order[i], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[i]


    return x_mesh_order

def coef_gamma(x_mesh, n, f):
    gamma_vec = np.zeros(n+1)
    func_val = np.zeros(n+1)

    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    for i in range(n+1):
        temp = 1
        for j in range(n+1):
            if j == i:
                continue
            else:
                temp = temp * (x_mesh[i]-x_mesh[j])
        gamma_vec[i] = temp

    gamma_vec = func_val/gamma_vec
    return gamma_vec, func_val

def coef_beta(x_mesh, n, f, flag):
    beta_vec = np.zeros(n+1)
    func_val = np.zeros(n+1)

    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    if flag == 1:
        beta_vec[0] = 1
        for i in range(n):
            beta_vec[i+1] = -beta_vec[i] * ((n - i) / i + 1)
    elif flag == 2:
        for i in range(n+1):
            beta_vec[i] = (-1)**i * np.sin(((2*i + 1) * np.pi) / (2*n + 2))
    else:
        for i in range(n+1):
            if i == 0 or i == n:
                beta_vec[i] = ((-1) ** i) * (1 / 2)
            else:
                beta_vec[i] = ((-1) ** i) * 1

    return beta_vec, func_val

def bary_1_interpolation(gamma_vec, x_mesh, x_values, y, n):
    """
    This function is implementing the Barycentric 1 form interpolation and evaluating the polynomial.
    The inputs for this function are:
    gamma_vec: Coefficient weights for p_(k-1)
    x_mesh: Given x-values
    x_values: x-values that are to be interpolated through/estimated
    y: The corresponding y-values associated with the mesh
    n: The length of the mesh minus 1

    Outputs:
    m_curr: The new Coefficient weights for p_k
    p_eval: the evaluated polynomial at the x_values
    """
    k = n+1

    m_curr = np.zeros(k)
    p = 1.0
    for i in range(n):
        t = x_mesh[i] - x_mesh[n]
        m_curr[i] = t * gamma_vec[i]
        p = -t * p

    m_curr[n] = p

    # Evaluating the polynomial given the weights calculated above
    p_eval = []
    for x in x_values:
        numer = 0
        denom = 0
        for j in range(k):
            if x != x_mesh[j]:
                term = m_curr[j] / (x - x_mesh[j])
                numer += term * y[j]
                denom += term

        p = numer / denom
        p_eval.append(p)

    np.array(p_eval)

    return m_curr, p_eval

def bary_2_interpolation(beta_vec, x_mesh, x_values, y, n):
    p_eval = []
    for x in x_values:
        numer = 0
        denom = 0
        for j in range(n+1):
            if x != x_mesh[j]:
                numer += (y[j] * beta_vec[j]) / (x - x_mesh[j])
                denom += beta_vec[j] / (x - x_mesh[j])

        p = numer / denom
        p_eval.append(p)

    np.array(p_eval)

    return p_eval

def newton_divdiff(x_mesh, x_values, f, n):
    func_val = np.zeros(n+1)
    d = np.zeros(n+1)

    """Computing the mesh values using the given function"""
    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    """Computing the vector of summation terms"""
    for i in range(n):
        omega_prime = 1.0
        for j in range(n+1):
            if i != j:
                omega_prime = omega_prime * (x_mesh[i] - x_mesh[j])
        d[i] = func_val[i] / omega_prime

    """Computing the coefficients"""
    p = x_mesh[0] - x_mesh[n]
    d_0 = d[0] / p
    s = d_0
    for i in range(1, n):
        t = (x_mesh[i] - x_mesh[n])
        d_i = d[i] / t
        p = t * p
        s = d_i + s

    d_n = ((-1) ** n) * (f(x_mesh[n]) / p)
    f_div = s + d_n

    # """Computing the polynomial evaluation using the divided difference coefficients just found"""
    #
    # # Initializing the first coefficient and holding the product term of x - x0 ...
    # p_eval = f_div[0]
    # product = 1.0
    #
    # for x in x_values:
    #     for i in range(1, n):
    #         product = product * (x - x_mesh[i-1])


    return f_div, func_val

def horner_interpolation(x_mesh, x_values, alpha, f_div, f, n):
    s = f_div
    p_eval = []
    for x in x_values:
        for i in range(n-1, 1, -1):
            s = s * (x - x_mesh[i]) + alpha[i]

        p_eval.append(s)

    return p_eval

# Testing functions
# x_mesh = np.array([1,2,3,4])
# y = np.array([10,26,58,112])
# x_values = np.array([1.5, 2.5, 3.5, 4.5])
# n=len(x_mesh)-1
# gamma_vec, func_val = coef_gamma(x_mesh, n, f)
# print(gamma_vec)
# print(func_val)
#
# m_curr, p_eval = bary_1_interpolation(gamma_vec, x_mesh, x_values, y, n)
# print(m_curr)
# print(p_eval)
#
# true_values = f(x_values)
# print(true_values)

# n = 20
# x_mesh = chebyshev_2(n)
# print(x_mesh)
# c, func_val = coef_beta(x_mesh, n, f_2, 3)
# print(c)
# print(func_val)
# x_values = np.linspace(-1, 1, 1000)
# ft = f_2(x_values)
#
#
# bary_2 = bary_2_interpolation(c, x_mesh, x_values, func_val, n)
# print(bary_2)
#
#
# plt.plot(x_mesh, func_val, '*')
# plt.plot(x_values, bary_2, '-')
# plt.plot(x_values, ft, '--')
# plt.grid(True)
# plt.show()





# x_mesh = np.array([-1, -0.5, 0, 0.5, 1])
# x_mesh_ordered = x_mesh_order(x_mesh, 3)
# print(x_mesh)
# print(x_mesh_ordered)