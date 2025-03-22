import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation


# ===================== Functions =====================

def f(x):
    x = np.asarray(x)
    #return np.sin(x)
    #return 3 * np.power(x, 3) + 4 * np.power(x, 2) + 2 * x + 1
    return 4 * np.power(x, 5) + 0.5 * np.power(x, 2) + 3 * x


def df(x):
    #return np.cos(x)
    #return 9 * np.power(x, 2) + 8 * x + 2
    return 20 * np.power(x, 4) + x + 3

def df2(x):
    #return -np.sin(x)
    #return 18 * x + 8
    return 80 * np.power(x, 3) + 1

function_description_for_title = {
    "f_1": r'$\sin(x)$',
    "f_2": r'$3x^3 + 4x^2 + 2x + 1$',
    "f_3": r'$4x^5 + \frac{1}{2}x^2 + 3x$'
}


# ===================== Settings =====================

# Bounds [a, b]
a = 0
b = 4 * np.pi

# Number of points to evaluate at
n = 100

# Number of Intervals for Barycentric (d + 1) Mesh points
#d = [2, 3, 5, 8, 12]
d = 6

# Precision type
dtype = np.float64

# Configuration for spline interpolation
piecewise = False

# Flag for type of Mesh Being Used
flag = 1

# Points to be tested
x_points = np.linspace(a, b, n)

# Graphing for f(x)
x_eval = np.linspace(a, b, 1000)

# Exact value of points
exact_value = f(x_points)
exact_plot = f(x_eval)

# ===================== Barycentric Interpolation =====================

# Initialize the Barycentric Interpolator
bary_interpolator = PolynomialInterpolation(a, b, d, dtype)

# Mesh and gamma coefficients for barycentric interpolation
bary_mesh = bary_interpolator.mesh_points(flag, piecewise)
y_mesh = f(bary_mesh)
gamma_vec, func_vals = bary_interpolator.gamma_coefficients(f)

# Perform barycentric interpolation
bary_eval = bary_interpolator.bary_1_interpolation(gamma_vec, x_points, func_vals)

# Plot for barycentric interpolation
plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label='f(x)')
plt.plot(x_points, bary_eval, 'b--', label='Barycentric 1')
plt.scatter(bary_mesh, y_mesh, label='Mesh Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Barycentric 1 Interpolation with Chebyshev Points of the Second Kind')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()