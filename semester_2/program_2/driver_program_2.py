import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation


# ===================== Functions =====================

def f(x):
    return np.sin(x)
    # return x**3 - 3*x**2 + 4


def df(x):
    return np.cos(x)
    # return 3*x**2 - 6*x


# ===================== Settings =====================

# Bounds [a, b]
a = 0
b = 2 * np.pi

# Number of points to evaluate at
n = 100

# Degree of Interpolating Polynomial
d = 3

# Precision type
dtype = np.float64

# ===================== Piecewise Interpolation =====================

# Configuration for piecewise interpolation
piecewise = True
hermite = True
flag = 1
M = [5]  # Number of Subintervals

# Initialize the dictionary for different number of intervals
interp_all = {m: [] for m in M}

# Points to be tested
x_points = np.linspace(a, b, n)
x_eval = np.linspace(a, b, 1000)

# Exact value of points tested
exact_value = f(x_points)
exact_plot = f(x_eval)

# Plot initialization
plt.figure(figsize=(12, 8))
plt.plot(x_eval, exact_plot, 'k', label='f(x)')

# Running the piecewise interpolation for different subintervals
for m in M:
    # Initialize the interpolator
    interpolator = PolynomialInterpolation(a, b, n, m, d, dtype)

    # Create local and global meshes
    interpolator.local_mesh(flag)
    interpolator.mesh_points(flag, piecewise)

    # Compute the divided differences
    interpolator.newton_divdiff(f, piecewise)

    # Initialize the vector for interpolated values
    interp_values = np.zeros_like(x_points)

    # Loop over x values and perform piecewise interpolation
    for i, x in enumerate(x_points):
        result = interpolator.piecewise_interpolation(x, f, df, flag, hermite)
        if result is not None:
            interp_values[i] = result

    # Append the interpolated values
    interp_all[m].append(interp_values)

    # Plot interpolated values and mesh points
    mesh_x = interpolator.x_mesh
    plt.plot(x_points, interp_values, linestyle='--', label=f'Interpolation (M = {m})')
    plt.scatter(mesh_x, f(mesh_x), s=20, label=f'Mesh Points (M = {m})')

# Finalize piecewise interpolation plot
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Piecewise Interpolation for Different Number of Subintervals M')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()

# ===================== Barycentric Interpolation =====================

piecewise = False
hermite = False
M = 0

bary_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

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

# ===================== Spline Interpolation =====================

piecewise = False
flag = 1
d = 6  # Degree for cubic spline interpolation

# Initialize the spline interpolator
spline_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

# Set up the mesh and compute spline coefficients
spline_mesh = spline_interpolator.mesh_points(flag, piecewise)
y_mesh = f(spline_mesh)
spline_order_mesh = spline_interpolator.ordered_mesh(flag)

# Compute second derivatives and spline coefficients
sp_second_deriv = spline_interpolator.spline_interpolation(f, flag)
sp_a, sp_b, sp_c, sp_d = spline_interpolator.get_spline_coefficients(sp_second_deriv, f, flag)

# Perform spline evaluation
sp = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    result = spline_interpolator.evaluate_spline(x, sp_a, sp_b, sp_c, sp_d, flag)
    sp[i] = result

# Plot for cubic spline interpolation
plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label='f(x)')
plt.plot(x_points, sp, 'b--', label='Cubic Spline')
plt.scatter(spline_mesh, y_mesh, label='Mesh Points')
plt.title('Cubic Spline Interpolation with Uniform Mesh')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()

# ===================== B-Spline =====================

piecewise = False
flag = 1
d = 12  # Degree for B-spline interpolation

# Initialize the B-spline interpolator
b_spline_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype)

# Perform B-spline interpolation
b_mesh = b_spline_interpolator.mesh_points(flag, piecewise)
b_y_mesh = f(b_mesh)
bp = np.zeros_like(x_points)
alpha = b_spline_interpolator.B_spline_interpolation(f, df)

# Evaluate B-spline at points
for i, x in enumerate(x_points):
    result = b_spline_interpolator.evaluate_B_Spline(x, alpha)
    bp[i] = result

# Plot for B-spline interpolation
plt.figure(figsize=(12, 8))
plt.plot(x_points, exact_value, 'k', label='f(x)')
plt.plot(x_points, bp, 'b--', label='B-Spline')
plt.scatter(b_mesh, b_y_mesh, label='Mesh Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()

# ===================== Task 2 =====================

x_mesh = [0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0]
y_values = [0.04, 0.05, 0.0682, 0.0801, 0.0940, 0.0981, 0.0912, 0.0857]
a = 0.5
b = 20
x_points = np.linspace(a, b, 40)

# Initialize Task 2 interpolator
task_2_interpolator = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values)

# Compute second derivatives and spline coefficients for task 2
t2_second_deriv = task_2_interpolator.spline_interpolation(f, flag)
t2_a, t2_b, t2_c, t2_d = task_2_interpolator.get_spline_coefficients(t2_second_deriv, f, flag)

# Estimate values for task 2
estimates = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    result = task_2_interpolator.evaluate_spline(x, t2_a, t2_b, t2_c, t2_d, flag)
    estimates[i] = result

print(estimates)


# Compute d(x, y)
def d(x, y):
    return np.exp(-x * y)


estimates_dt = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    result = d(x, estimates[i])
    estimates_dt[i] = result

print(estimates_dt)


# Define f_2 for Task 2 and estimate its values
def f_2(x, y, y_prime):
    return y + x * y_prime


estimates_f = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    y_prime = task_2_interpolator.evaluate_spline_derivative(x, t2_a, t2_b, t2_c, t2_d, flag)
    result = f_2(x, estimates[i], y_prime)
    estimates_f[i] = result

print(estimates_f)

# Initialize Task 2 interpolator again if needed
M = 7
d = 1
task_2_interpolator_piece = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values)

div_coeff = task_2_interpolator_piece.newton_divdiff(f, piecewise=True, specify=True)

estimates_piecewise = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    estimates_piecewise[i] = task_2_interpolator_piece.piecewise_interpolation(x, f, df, flag, hermite=False, specify=True)

print(estimates_piecewise)

plt.figure(figsize=(12, 8))
plt.plot(x_points, estimates_piecewise, label='piecewise polynomial interpolation')
plt.scatter(x_mesh, y_values, s=20, label='mesh points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()


