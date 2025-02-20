import copy
import numpy as np
import matplotlib.pyplot as plt

# def f(x):
#     # f(x)=4+3x+2x^2+x^3
#     return 4 + 3 * x + 2 * np.power(x,2) + np.power(x,3)
#
# def f_2(x):
#     #return abs(x) + 0.5 * x - x ** 2
#     return (x - 2)**2

def chebyshev_points(a, b, n, flag, dtype=np.float32):
    """
    Function to create Chebyshev Points of the First kind
    Inputs:
        n: number of points needed
        flag: Flag to indicate which Chebyshev Points should be created
            1 = Uniform Mesh
            2 = Chebyshev Points of the First Kind
            3 = Chebyshev Points of the Second Kind
    Outputs:
        x_mesh: mesh points with desired type
    """
    # Initializing the mesh
    x_mesh = np.zeros(n+1)


    if flag == 1:
        x_mesh = np.linspace(a, b, n+1)
    # Looping over to create the mesh points for Chebyshev Points of the First Kind
    elif flag == 2:
        for i in range(n+1):
            x_mesh[i] = 0.5 * (b-a) * np.cos(((2*i + 1) * np.pi) / (2 * n + 2)) + 0.5*(b+a)

    # Looping over to create the mesh points for Chebyshev Points of the Second Kind
    elif flag == 3:
        for i in range(n+1):
            x_mesh[i] = 0.5 * (b-a) * np.cos((i * np.pi)/ n) + 0.5 * (b+a)

    return x_mesh.astype(dtype)

def x_mesh_order(x_mesh, flag):
    """
    Function to order a given mesh in either increasing, decreasing, or Leja order
    Inputs:
        x_mesh: The vector of points to be sorted
        flag: Flag to indicate which ordering is to be used
            1 = Decreasing order of values in mesh
            2 = Increasing order of values in mesh
            3 = Leja order of values in mesh
    Outputs:
        x_mesh_order: Points in order
    """

    # Initializing terms and the ordered set of terms
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

    # Leja ordering of x-values in mesh
    elif flag == 3:
        # Picking x_0 as the max by getting the index of the max through argmax and then reordering terms
        x_max_index = np.argmax(np.abs(x_mesh_order))
        x_mesh_order[0], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[0]

        # Looping over Rest of the values in the mesh starting from index 1 to n-1
        for i in range(1, n):
            # Initializing the product to 1
            product = np.ones(n)

            # Looping over the remaining values needed for the product, i to n-1
            for j in range(i, n):
                for k in range(i): # Looping over the already selected points from before, i.e. 0 to i-1
                    product[j] *= abs(x_mesh_order[j] - x_mesh_order[k])


            # Indexing through to find the next point with max product and add i to get correct index
            # Adding i is needed as this will adjust the index in the subarray to the original array location
            x_max_index = np.argmax(np.abs(product[i:])) + i

            # Reordering mesh based on the point with max product
            x_mesh_order[i], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[i]


    return x_mesh_order

def coef_gamma(x_mesh, n, f, dtype=np.float32):
    """
    Function to calculate the gamma coefficients and have an evaluation of the function at the mesh points
    Inputs:
        x_mesh: The mesh points
        n: number of points needed
        f: Function to be evaluated/to evaluate the points
    Outputs:
        gamma_vec: The gamma coefficients stored in a vector
        func_val: The evaluated function value at the mesh points
    """

    gamma_vec = np.ones(n+1)
    func_val = np.zeros(n+1)

    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                gamma_vec[i] *= (x_mesh[i] - x_mesh[j])

    gamma_vec = 1/gamma_vec
    return gamma_vec.astype(dtype), func_val

def coef_beta(x_mesh, n, f, flag, dtype=np.float32):
    """
    Function to calculate the beta coefficients for Barycentric 2 using either the recursive formula to
    get the coefficients, Chebyshev points of the First Kind, or Chebyshev Points of the Second Kind. This
    function will also evaluate the function at the mesh points
    Inputs:
        x_mesh: The mesh points
        n: number of points needed
        f: Function to be evaluated/to evaluate the points
        flag: Flag to indicate which beta coefficients should be created
            1 = Recursive style without Chebyshev
            2 = Using Chebyshev Points of the First Kind
            3 = Using Chebyshev Points of the Second Kind
    Outputs:
        beta_vec: The beta coefficients stored in a vector
        func_val: The evaluated function value at the mesh points
    """

    beta_vec = np.zeros(n+1)
    func_val = np.zeros(n+1)

    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    # Using Uniform Mesh
    if flag == 1:
        beta_vec[0] = 1
        for i in range(n):
            beta_vec[i+1] = -beta_vec[i] * ((n - i) / i + 1)

    # Chebyshev Point of the First Kind
    elif flag == 2:
        for i in range(n+1):
            beta_vec[i] = (-1)**i * np.sin(((2*i + 1) * np.pi) / (2*n + 2))

    # Chebyshev Points of the Second Kind
    else:
        for i in range(n+1):
            if i == 0 or i == n:
                beta_vec[i] = ((-1) ** i) * (1 / 2)
            else:
                beta_vec[i] = ((-1) ** i) * 1

    return beta_vec.astype(dtype), func_val

def bary_1_interpolation(gamma_vec, x_mesh, x_values, y, n, dtype=np.float32):
    """
    This function is implementing the Barycentric 1 form interpolation and evaluating the polynomial.
    Inputs:
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

    # m_curr = np.zeros(k)
    # p = 1.0
    # for i in range(n):
    #     t = x_mesh[i] - x_mesh[n]
    #     m_curr[i] = t * gamma_vec[i]
    #     p = -t * p
    #
    # m_curr[n] = p

    # omega = 1
    # for x in x_values:
    #     for i in range(k):
    #         if x != x_mesh[i]:
    #             omega *= x - x_mesh[i]

    # Evaluating the polynomial given the weights calculated above
    # p_eval = []
    # for x in x_values:
    #     #numer = 0
    #     #denom = 0
    #     term = 0
    #     #### FIX THIS ####
    #     for j in range(k):
    #         if x != x_mesh[j]:
    #             #term = m_curr[j] / (x - x_mesh[j])
    #             term += y[j] * gamma_vec[j] / (x - x_mesh[j])
    #             #numer += term * y[j]
    #             #denom += term
    #
    #
    #     p = term * omega
    #     p_eval.append(p)
    #
    # np.array(p_eval)

    p_eval = []
    for j in range(len(x_values)):
        omega = np.prod(x_values[j] - x_mesh)

        term = 0
        for i in range(k):
            term += (y[i] * gamma_vec[i]) / (x_values[j] - x_mesh[i])

        p = omega * term
        p_eval.append(p)

    p_eval = np.array(p_eval, dtype=dtype)
    return p_eval

def bary_2_interpolation(beta_vec, x_mesh, x_values, y, n, dtype=np.float32):
    p_eval = []
    for x in x_values:
        numer = 0
        denom = 0
        for j in range(n+1):
            if x != x_mesh[j]:
                numer += (y[j] * beta_vec[j]) / (x - x_mesh[j])
                denom += beta_vec[j] / (x - x_mesh[j])
            else:
                idx = np.where(x_mesh[j] == x)[0][0]
                p_eval.append(y[idx])
                continue


        p = numer / denom
        p_eval.append(p)

    p_eval = np.array(p_eval, dtype = dtype)

    return p_eval

def newton_divdiff(x_mesh, f, n, dtype=np.float32):
    func_val = np.zeros(n+1)

    """Computing the mesh values using the given function"""
    for i in range(n+1):
        func_val[i] = f(x_mesh[i])

    div_coeff = copy.deepcopy(func_val)
    for i in range(1, n+1):
        for j in range(n, i-1, -1):
            div_coeff[j] = (div_coeff[j] - div_coeff[j-1]) / (x_mesh[j] - x_mesh[j-i])

    return func_val, div_coeff.astype(dtype)

def horner_interpolation(x_mesh, x_values, div_coeff, n, dtype=np.float32):
    alpha = copy.deepcopy(div_coeff)
    s = alpha[-1]
    p_eval = []
    for x in x_values:
        for i in range(n-1, 1, -1):
            s = s * (x - x_mesh[i]) + alpha[i]

        p_eval.append(s)

    p_eval = np.array(p_eval, dtype=dtype)

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
## Testing Barycentric 1 function
# m_curr, p_eval = bary_1_interpolation(gamma_vec, x_mesh, x_values, y, n)
# print(m_curr)
# print(p_eval)
#
# true_values = f(x_values)
# print(true_values)

# Testing Barycentric 2 example given in class
# n = 1
# x_mesh = chebyshev_points(n, flag=2, dtype=np.float32)
# print(x_mesh)
# c, func_val = coef_beta(x_mesh, n, f_2, 3)
# print(c)
# print(func_val)
# x_values = np.linspace(-1, 1, 1000)
# ft = f_2(x_values)
# #
# #
# bary_2 = bary_2_interpolation(c, x_mesh, x_values, func_val, n)
# print(bary_2)
#
#
# plt.plot(x_mesh, func_val, '*')
# plt.plot(x_values, bary_2, '-')
# plt.plot(x_values, ft, '--')
# plt.grid(True)
# plt.show()

# Testing Newton Divided Difference Table
# x_mesh = [1, 2, 4, 7, 8]
# def f_3(x):
#     return x**3 - 4*x
#
# n = len(x_mesh) - 1
#
# func_val, div = newton_divdiff(x_mesh, f_3, n, dtype=np.float64)
# print(func_val)
# print(div)

# Testing Leja Ordering to make sure ordering is correct
# x_mesh = np.array([-1, -0.5, 0, 0.5, 1, 2, -2, 5, -3, 4, 10, -10])
# x_mesh_ordered = x_mesh_order(x_mesh, 3)
# print(x_mesh)
# print(x_mesh_ordered)