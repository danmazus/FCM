import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation


# ===================== Task 2 =======================


# =================== Functions ======================

def f(x):
    return np.sin(x)
    # return x**3 - 3*x**2 + 4


def df(x):
    return np.cos(x)
    # return 3*x**2 - 6*x

# ==================== Settings ======================

# Setting the mesh and y-values for the mesh
x_mesh = [0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0]
y_values = [0.04, 0.05, 0.0682, 0.0801, 0.0940, 0.0981, 0.0912, 0.0857]

# Bounds for interpolation
a = 0.5
b = 20

# Initialize n needed to setup interpolator
n = 40

# Number of Intervals for Piecewise Polynomial Interpolation
M = 6

# Degree for Piecewise Polynomial Interpolation
d = 3

# Type of mesh being used
flag = 1

# Specify Precision to be used
dtype = np.float64

# Points to be tested
x_points = np.linspace(a, b, n)


# ==================== Cubic Spline Interpolation ======================

# Initialize Task 2 interpolator
task_2_interpolator = PolynomialInterpolation(a, b, d, dtype, M, x_mesh, y_values)

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
def dt(x, y):
    return np.exp(-x * y)


estimates_dt = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    result = dt(x, estimates[i])
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



# ==================== Piecewise Interpolation ======================

# Initialize Task 2 interpolator again if needed
task_2_interpolator_piece = PolynomialInterpolation(a, b, d, dtype, M, x_mesh, y_values)

# Divided Difference Coefficients for Piecewise Interpolation
div_coeff = task_2_interpolator_piece.newton_divdiff(f, piecewise=True, specify=True)

# Computing the Piecewise Estimates
estimates_piecewise = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    estimates_piecewise[i] = task_2_interpolator_piece.piecewise_interpolation(x, f, df, flag, hermite=False, specify=True)

print(estimates_piecewise)

estimates_piecewise_dt = np.zeros_like(x_points)
for i, x in enumerate(x_points):
    estimates_piecewise_dt[i] = dt(x, estimates_piecewise[i])

print(estimates_piecewise_dt)

plt.figure(figsize=(12, 8))
plt.plot(x_points, estimates_piecewise, label='piecewise polynomial interpolation')
#plt.plot(x_points, estimates_dt, label='derivative of piecewise polynomial interpolation')
plt.scatter(x_mesh, y_values, s=20, label='mesh points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(fontsize='x-small')
plt.grid(True)
plt.show()