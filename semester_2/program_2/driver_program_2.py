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


"""PIECEWISE INTERPOLATION"""
# Which point type to use
flag = 1

# Setting States of Interpolation
piecewise = True
hermite = True

# Number of Subintervals
M = [5]

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




"""Barycentric Interpolation"""
piecewise=False
hermite=False
M = 0

bary_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

bary_mesh = bary_interpolator.mesh_points(flag, piecewise)
y_mesh = f(bary_mesh)
gamma_vec, func_vals = bary_interpolator.gamma_coefficients(f)
bary_eval = bary_interpolator.bary_1_interpolation(gamma_vec, x_points, func_vals)

plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label = 'f(x)')
plt.plot(x_points, bary_eval, 'b--', label = 'Barycentric 1')
plt.scatter(bary_mesh, y_mesh, label = 'Mesh Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Barycentric 1 Interpolation with Chebyshev Points of the Second Kind')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()


"""Spline Interpolation"""
piecewise=False
flag = 1

# Number of Intervals (d + 1 mesh points) used as piecewise is False
d = 3

# Setup the Interpolator
spline_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

# Setting up the mesh
spline_mesh = spline_interpolator.mesh_points(flag, piecewise)
y_mesh = f(spline_mesh)
spline_order_mesh = spline_interpolator.ordered_mesh(flag)


sp_second_deriv  = spline_interpolator.spline_interpolation(f, df, flag)
sp_a, sp_b, sp_c, sp_d = spline_interpolator.get_spline_coefficients(sp_second_deriv, f, flag)

sp = np.zeros_like(x_points)

for i, x in enumerate(x_points):
    result = spline_interpolator.evaluate_spline(x, sp_a, sp_b, sp_c, sp_d, flag)
    sp[i] = result

print(sp)

plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label = 'f(x)')
plt.plot(x_points, sp, 'b--', label = 'Cubic Spline')
plt.scatter(spline_mesh, y_mesh, label = 'Mesh Points')
plt.title('Cubic Spline Interpolation with Uniform Mesh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()
