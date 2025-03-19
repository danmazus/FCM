import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation


def f(x):
    return np.sin(x)
    #return x**3 - 3*x**2 + 4

def df(x):
    return np.cos(x)
    #return 3*x**2 - 6*x


# Creating Bounds [a, b]
a = 0
b = 2*np.pi

# Number of points to evaluate at
n = 100

# Degree of Interpolating Polynomial
d = 3

# What precision to use
dtype = np.float64

# Which point type to use
flag = 3

# Setting States of Interpolation
piecewise = True
hermite = True

# Number of Subintervals
M = [6]

# Initializing dictionary for different number of intervals
interp_all = {m: [] for m in M}

# Points to be tested
x_points = np.linspace(a, b, n)
x_eval = np.linspace(a, b, 1000)

# Exact value of points tested
exact_value = f(x_points)
exact_plot = f(x_eval)

# Initializing Figure for Showing g_d(x) vs f(x)
plt.figure(figsize=(12, 8))
plt.plot(x_eval, exact_plot, 'k', label = 'f(x)')

# Running over all number of subintervals
for m in M:
    # Initializing State of Interpolator
    interpolator = PolynomialInterpolation(a, b, n, m, d, dtype)

    # Creating the local and global meshes
    interpolator.local_mesh(flag)
    interpolator.mesh_points(flag, piecewise)

    # Computing the divided differences
    interpolator.newton_divdiff(f, piecewise)

    # Initializing the interpolated values vector
    interp_values = np.zeros_like(x_points)

    # Looping over the x values to be tested
    for i, x in enumerate(x_points):
        # Running the piecewise interpolation
        result = interpolator.piecewise_interpolation(x, f, df, flag, hermite)
        if result is not None:
            interp_values[i] = result

    # Appending interpolated values to the dictionary for given subinterval
    interp_all[m].append(interp_values)

    # Setting variable for the mesh
    mesh_x = interpolator.x_mesh

    # Plotting the interpolated values and x values that were interpolated
    plt.plot(x_points, interp_values, linestyle = '--', label = f'Interpolation (M = {m})')

    # Plotting the mesh points
    plt.scatter(mesh_x, f(mesh_x), s=20, label = f'Mesh Points (M = {m})')

# Finishing the plot for interpolation graph
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Interpolation for Different Number of Subintervals M')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()

# # Create the interpolater
# interpolater = PolynomialInterpolation(a, b, n, M, d, dtype)
#
# # Generate the mesh
# interpolater.local_mesh(flag)
# interpolater.mesh_points(flag, piecewise)
#
# # Computing the divided differences
# interpolater.newton_divdiff(f, df, piecewise, hermite)
#
# # Create the points to evaluate at
# x_eval = np.linspace(a, b, 100)
#
# # Get the interpolated values for x_eval
# interp_values = np.zeros_like(x_eval)
# for i, x in enumerate(x_eval):
#     result = interpolater.piecewise_interpolation(x, flag, hermite)
#     if result is not None:
#         interp_values[i] = result
#
#
# # The exact function values of x
# exact_values = f(x_eval)
#
#
# # Getting error
# max_error = np.max(np.abs(exact_values - interp_values))
# print(f'Maximum interpolation error: {max_error}')
#
# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(x_eval, exact_values, 'b', label = 'f(x)')
# plt.plot(x_eval, interp_values, 'r--', label = 'Interpolation')
# plt.scatter(interpolater.x_mesh, f(interpolater.x_mesh), c = 'k', s = 30, label = 'Mesh Points')
# plt.legend(loc = 'best')
# plt.grid(True)
# plt.title('Piecewise Hermite Interpolation' if hermite else 'Piecewise Interpolation')
# plt.show()


# flag = 1
# n = 20
#
# spline_interpolater = PolynomialInterpolation(a, b, n, M, d, dtype)
#
# spline_interpolater.mesh_points(flag, piecewise=False)
#
# spline_coeffs = spline_interpolater.cubic_spline(f, boundary=2)
#
# x_plot = np.linspace(a, b, 1000)
# y_exact = f(x_plot)
# y_spline = np.array([spline_interpolater.evaluate_spline(x, spline_coeffs) for x in x_plot])
#
# plt.figure(figsize=(10, 6))
# plt.plot(x_plot, y_exact, 'b', label = 'f(x)')
# plt.plot(x_plot, y_spline, 'r--', label = 'Spline')
# plt.scatter(spline_interpolater.x_mesh, f(spline_interpolater.x_mesh), c = 'k', label = 'Mesh Points')
# plt.legend(loc = 'best')
# plt.grid(True)
# plt.title('Cubic Spline Interpolation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
# x_derivative = np.linspace(a, b, 100)
# y_derivative_exact = df(x_derivative)
# y_derivative_spline = np.array([spline_interpolater.evaluate_spline_deriv(x, spline_coeffs, order=1) for x in x_derivative])
# plt.figure(figsize=(10, 6))
# plt.plot(x_derivative, y_derivative_exact, 'b', label = 'f(x)')
# plt.plot(x_derivative, y_derivative_spline, 'r--', label = 'Spline')
# plt.legend(loc = 'best')
# plt.grid(True)
# plt.title('Derivative')
# plt.xlabel('x')
# plt.ylabel('dy/dx')
# plt.show()

# Barycentric Interpolation
piecewise=False
hermite=False
M = 0

bary_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

bary_mesh = bary_interpolator.mesh_points(flag, piecewise)
gamma_vec, func_vals = bary_interpolator.gamma_coefficients(f)
bary_eval = bary_interpolator.bary_1_interpolation(gamma_vec, x_points, func_vals)

plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label = 'f(x)')
plt.plot(x_points, bary_eval, 'b--', label = 'bary_1_interpolation')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Barycentric 1 Interpolation with Chebyshev Points of the Second Kind')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()
