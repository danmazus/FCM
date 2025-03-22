import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation

# def g(x):
#     a, b, c, d = 2, -3, 1, 4
#     return a*x**3 + b*x**2 + c*x + d
#
# def piecewise_cubic(x, x_mesh):
#     for i in range(len(x_mesh) - 1):
#         if x_mesh[i] <= x <= x_mesh[i + 1]:
#             a, b, c, d = i+1, -2*i, 3, -1
#             return a*x**3 + b*x**2 + c*x + d
#     return 0
#
# a, b = 0, 10
# n = 5
# M, d = 0, 3
# dtype = np.float64
#
# x_mesh = np.linspace(a, b, n)
# y_values_g = g(x_mesh)
# y_values_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_mesh])
#
# # Initialize the Interpolator
# interpolator_g = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_g)
# interpolator_piecewise = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_piece)
#
# # Compute the Cubic Spline
# second_deriv_g = interpolator_g.spline_interpolation(g, flag=1)
# a_g, b_g, c_g, d_g = interpolator_g.get_spline_coefficients(second_deriv_g, g, flag=1)
#
# # Compute the Cubic Spline for Piecewise Cubic
# second_deriv_piece = interpolator_piecewise.spline_interpolation(piecewise_cubic, flag=1)
# a_p, b_p, c_p, d_p = interpolator_piecewise.get_spline_coefficients(second_deriv_piece, piecewise_cubic, flag=1)
#
# # Test Points
# x_test = np.linspace(a, b, 100)
#
# # Evaluation
# gp = np.array([interpolator_g.evaluate_spline(x, a_g, b_g, c_g, d_g, flag=1) for x in x_test])
# sp = np.array([interpolator_piecewise.evaluate_spline(x, a_p, b_p, c_p, d_p, flag=1) for x in x_test])
#
# # Exact
# exact_g = g(x_test)
# exact_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_test])
#
# # Max Errors
# print("Max Error for cubic polynomial test: ", np.max(np.abs(gp - exact_g)))
# print("Max Error for piecewise cubic test: ", np.max(np.abs(sp - exact_piece)))
#
# # Plot the results
# plt.figure(figsize=(12, 6))
#
# # Plot for the cubic polynomial g(x)
# plt.subplot(1, 2, 1)
# plt.plot(x_test, exact_g, label='Ground Truth g(x)', color='blue')
# plt.plot(x_test, gp, label='Spline Approximation', linestyle='dashed', color='orange')
# plt.scatter(x_mesh, y_values_g, label='Mesh Points', color='black')
# plt.title('Cubic Polynomial Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
#
# # Plot for the piecewise cubic
# plt.subplot(1, 2, 2)
# plt.scatter(x_mesh, y_values_piece, s=20, label='Mesh Points', color='black')
# plt.plot(x_test, exact_piece, label='Ground Truth Piecewise', color='green')
# plt.plot(x_test, sp, label='Spline Approximation', linestyle='dashed', color='red')
# plt.title('Piecewise Cubic Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


# Define a cubic polynomial function
# def g(x):
#     a, b, c, d = 1, 1, 1, 4
#     return a*x**3 + b*x**2 + c*x + d
#
# # Define a piecewise cubic function with different cubic polynomials on each interval
# def piecewise_cubic(x, x_mesh):
#     for i in range(len(x_mesh) - 1):
#         if x_mesh[i] <= x < x_mesh[i + 1] or (i == len(x_mesh) - 2 and x == x_mesh[i + 1]):
#             a, b, c, d = i+1, -2*i, 3, -1
#             return a*x**3 + b*x**2 + c*x + d
#     return 0  # Default return if x is outside the mesh range
#
# # Set up the domain and mesh
# a, b = -5, 5  # Domain endpoints
# n = 10       # Number of mesh points
# M, d = n-1, 3   # Parameters for interpolation class (M likely boundary condition, d is degree)
# dtype = np.float64
#
# # Create mesh points
# x_mesh = np.linspace(a, b, n)
#
# # Evaluate functions at mesh points
# y_values_g = g(x_mesh)
# y_values_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_mesh])
#
# # Initialize the interpolators
# interpolator_g = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_g)
# interpolator_piecewise = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_piece)
#
# # Create a wrapper for the piecewise function to handle the x_mesh parameter
# def piecewise_wrapper(x):
#     return piecewise_cubic(x, x_mesh)
#
# # Compute the cubic spline for g(x)
# second_deriv_g = interpolator_g.spline_interpolation(g, flag=1)
# a_g, b_g, c_g, d_g = interpolator_g.get_spline_coefficients(second_deriv_g, g, flag=1)
#
# # Compute the cubic spline for the piecewise cubic function
# second_deriv_piece = interpolator_piecewise.spline_interpolation(piecewise_wrapper, flag=1)
# a_p, b_p, c_p, d_p = interpolator_piecewise.get_spline_coefficients(second_deriv_piece, piecewise_wrapper, flag=1)
#
# # Create test points for evaluation
# num_test_points = 1000  # More points for smoother visualization
# x_test = np.linspace(a, b, num_test_points)
#
# # Evaluate the spline approximations at test points
# gp = np.array([interpolator_g.evaluate_spline(x, a_g, b_g, c_g, d_g, flag=1) for x in x_test])
# sp = np.array([interpolator_piecewise.evaluate_spline(x, a_p, b_p, c_p, d_p, flag=1) for x in x_test])
#
# # Evaluate the exact functions at test points
# exact_g = g(x_test)
# exact_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_test])
#
# # Calculate errors
# error_g = np.abs(gp - exact_g)
# error_piece = np.abs(sp - exact_piece)
# max_error_g = np.max(error_g)
# max_error_piece = np.max(error_piece)
#
# # Print the maximum errors
# print(f"Max Error for cubic polynomial test: {max_error_g:.2e}")
# print(f"Max Error for piecewise cubic test: {max_error_piece:.2e}")
#
# # Plot the results
# plt.figure(figsize=(15, 10))
#
# # Plot for the cubic polynomial g(x)
# plt.subplot(2, 2, 1)
# plt.plot(x_test, exact_g, label='Ground Truth g(x)', color='blue')
# plt.plot(x_test, gp, label='Spline Approximation', linestyle='dashed', color='orange')
# plt.scatter(x_mesh, y_values_g, label='Mesh Points', color='black')
# plt.title('Cubic Polynomial Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Plot for the piecewise cubic
# plt.subplot(2, 2, 2)
# plt.plot(x_test, exact_piece, label='Ground Truth Piecewise', color='green')
# plt.plot(x_test, sp, label='Spline Approximation', linestyle='dashed', color='red')
# plt.scatter(x_mesh, y_values_piece, s=30, label='Mesh Points', color='black')
# plt.title('Piecewise Cubic Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Plot the errors
# plt.subplot(2, 2, 3)
# plt.semilogy(x_test, error_g + 1e-16)  # Add small value to handle zeros
# plt.title(f'Error for Cubic Polynomial (Max: {max_error_g:.2e})')
# plt.xlabel('x')
# plt.ylabel('Error (log scale)')
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 4)
# plt.semilogy(x_test, error_piece + 1e-16)  # Add small value to handle zeros
# plt.title(f'Error for Piecewise Cubic (Max: {max_error_piece:.2e})')
# plt.xlabel('x')
# plt.ylabel('Error (log scale)')
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('cubic_spline_verification.png')
# plt.show()
#
# # Additional verification
# print("\nVerification Summary:")
# print(f"- Cubic polynomial g(x) = 2x³ - 3x² + x + 4")
# print(f"- Error should be close to machine precision: {max_error_g:.2e}")
# print(f"- Piecewise cubic function with different cubics on each interval")
# print(f"- Error should be close to machine precision: {max_error_piece:.2e}")
#
# # Check if test passes (tolerance based on floating-point precision)
# tolerance = 1e-10
# if max_error_g < tolerance and max_error_piece < tolerance:
#     print("\nTEST PASSED: Cubic spline correctly reproduces both functions!")
# else:
#     print("\nTEST FAILED: Errors exceed expected tolerance.")
#     if max_error_g >= tolerance:
#         print(f"- Cubic polynomial test failed: error = {max_error_g:.2e}")
#     if max_error_piece >= tolerance:
#         print(f"- Piecewise cubic test failed: error = {max_error_piece:.2e}")
#
# # Define a quadratic polynomial function (degree 2)
# def g(x):
#     a, b, c = 2, -3, 4  # ax² + bx + c
#     return a*x**2 + b*x + c
#
# # Define a piecewise quadratic function with different quadratics on each interval
# def piecewise_quadratic(x, x_mesh):
#     for i in range(len(x_mesh) - 1):
#         if x_mesh[i] <= x < x_mesh[i + 1] or (i == len(x_mesh) - 2 and x == x_mesh[i + 1]):
#             a, b, c = i+1, -2*i, 3  # ax² + bx + c (different for each interval)
#             return a*x**2 + b*x + c
#     return 0  # Default return if x is outside the mesh range
#
# # Set up the domain and mesh
# a, b = 0, 10  # Domain endpoints
# n = 5         # Number of mesh points
# M, d = 0, 3   # Parameters for interpolation class
# dtype = np.float64
#
# # Create mesh points
# x_mesh = np.linspace(a, b, n)
#
# # Evaluate functions at mesh points
# y_values_g = g(x_mesh)
# y_values_piece = np.array([piecewise_quadratic(x, x_mesh) for x in x_mesh])
#
# # Initialize the interpolators
# interpolator_g = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_g)
# interpolator_piecewise = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_piece)
#
# # Create a wrapper for the piecewise function to handle the x_mesh parameter
# def piecewise_wrapper(x):
#     return piecewise_quadratic(x, x_mesh)
#
# # Compute the cubic spline for g(x)
# second_deriv_g = interpolator_g.spline_interpolation(g, flag=1)
# a_g, b_g, c_g, d_g = interpolator_g.get_spline_coefficients(second_deriv_g, g, flag=1)
#
# # Compute the cubic spline for the piecewise quadratic function
# second_deriv_piece = interpolator_piecewise.spline_interpolation(piecewise_wrapper, flag=1)
# a_p, b_p, c_p, d_p = interpolator_piecewise.get_spline_coefficients(second_deriv_piece, piecewise_wrapper, flag=1)
#
# # Create test points for evaluation
# num_test_points = 1000  # More points for smoother visualization
# x_test = np.linspace(a, b, num_test_points)
#
# # Evaluate the spline approximations at test points
# gp = np.array([interpolator_g.evaluate_spline(x, a_g, b_g, c_g, d_g, flag=1) for x in x_test])
# sp = np.array([interpolator_piecewise.evaluate_spline(x, a_p, b_p, c_p, d_p, flag=1) for x in x_test])
#
# # Evaluate the exact functions at test points
# exact_g = g(x_test)
# exact_piece = np.array([piecewise_quadratic(x, x_mesh) for x in x_test])
#
# # Calculate errors
# error_g = np.abs(gp - exact_g)
# error_piece = np.abs(sp - exact_piece)
# max_error_g = np.max(error_g)
# max_error_piece = np.max(error_piece)
#
# # Print the maximum errors
# print(f"Max Error for quadratic polynomial test: {max_error_g:.2e}")
# print(f"Max Error for piecewise quadratic test: {max_error_piece:.2e}")
#
# # Plot the results
# plt.figure(figsize=(15, 10))
#
# # Plot for the quadratic polynomial g(x)
# plt.subplot(2, 2, 1)
# plt.plot(x_test, exact_g, label='Ground Truth g(x)', color='blue')
# plt.plot(x_test, gp, label='Spline Approximation', linestyle='dashed', color='orange')
# plt.scatter(x_mesh, y_values_g, label='Mesh Points', color='black')
# plt.title('Quadratic Polynomial Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Plot for the piecewise quadratic
# plt.subplot(2, 2, 2)
# plt.plot(x_test, exact_piece, label='Ground Truth Piecewise', color='green')
# plt.plot(x_test, sp, label='Spline Approximation', linestyle='dashed', color='red')
# plt.scatter(x_mesh, y_values_piece, s=30, label='Mesh Points', color='black')
# plt.title('Piecewise Quadratic Approximation')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True, alpha=0.3)
#
# # Plot the errors
# plt.subplot(2, 2, 3)
# plt.semilogy(x_test, error_g + 1e-16)  # Add small value to handle zeros
# plt.title(f'Error for Quadratic Polynomial (Max: {max_error_g:.2e})')
# plt.xlabel('x')
# plt.ylabel('Error (log scale)')
# plt.grid(True, alpha=0.3)
#
# plt.subplot(2, 2, 4)
# plt.semilogy(x_test, error_piece + 1e-16)  # Add small value to handle zeros
# plt.title(f'Error for Piecewise Quadratic (Max: {max_error_piece:.2e})')
# plt.xlabel('x')
# plt.ylabel('Error (log scale)')
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig('quadratic_spline_verification.png')
# plt.show()
#
# # Additional verification
# print("\nVerification Summary:")
# print(f"- Quadratic polynomial g(x) = 2x² - 3x + 4")
# print(f"- Error should be small but not necessarily machine precision: {max_error_g:.2e}")
# print(f"- Piecewise quadratic function with different quadratics on each interval")
# print(f"- Error should be small but not necessarily machine precision: {max_error_piece:.2e}")
#
# # Check if test passes (using a more relaxed tolerance for quadratics)
# tolerance = 1e-2  # More relaxed tolerance since we don't expect perfect reproduction
# if max_error_g < tolerance and max_error_piece < tolerance:
#     print("\nTEST PASSED: Cubic spline adequately approximates both functions!")
# else:
#     print("\nTEST FAILED: Errors exceed expected tolerance.")
#     if max_error_g >= tolerance:
#         print(f"- Quadratic polynomial test failed: error = {max_error_g:.2e}")
#     if max_error_piece >= tolerance:
#         print(f"- Piecewise quadratic test failed: error = {max_error_piece:.2e}")

import numpy as np
import matplotlib.pyplot as plt

# Define a cubic polynomial g(x) = ax^3 + bx^2 + cx + d
def g(x):
    a, b, c, d = 1, 1, 1, 4  # Arbitrary coefficients
    return a*x**3 + b*x**2 + c*x + d

# Define a piecewise cubic function with different cubic polynomials on each interval
def piecewise_cubic(x, x_mesh):
    for i in range(len(x_mesh) - 1):
        if x_mesh[i] <= x < x_mesh[i + 1] or (i == len(x_mesh) - 2 and x == x_mesh[i + 1]):
            a, b, c, d = i+1, -2*i, 3, -1
            return a*x**3 + b*x**2 + c*x + d
    return 0  # Default return if x is outside the mesh range

# Set up the domain and mesh
a, b = -5, 5  # Domain endpoints
n = 10       # Number of mesh points
M, d = n-1, 6   # Parameters for interpolation class (M likely boundary condition, d is degree)
dtype = np.float64

# Create mesh points
x_mesh = np.linspace(a, b, n)

# Evaluate functions at mesh points
y_values_g = g(x_mesh)
y_values_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_mesh])

# Initialize the interpolators
interpolator_g = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_g)
interpolator_piecewise = PolynomialInterpolation(a, b, n, M, d, dtype, x_mesh, y_values_piece)

# Create a wrapper for the piecewise function to handle the x_mesh parameter
def piecewise_wrapper(x):
    return piecewise_cubic(x, x_mesh)

# Compute the cubic spline for g(x)
second_deriv_g = interpolator_g.spline_interpolation(g, flag=1)
a_g, b_g, c_g, d_g = interpolator_g.get_spline_coefficients(second_deriv_g, g, flag=1)

# Compute the cubic spline for the piecewise cubic function
second_deriv_piece = interpolator_piecewise.spline_interpolation(piecewise_wrapper, flag=1)
a_p, b_p, c_p, d_p = interpolator_piecewise.get_spline_coefficients(second_deriv_piece, piecewise_wrapper, flag=1)

# Create test points for evaluation
num_test_points = 1000  # More points for smoother visualization
x_test = np.linspace(a, b, num_test_points)

# Evaluate the spline approximations at test points
gp = np.array([interpolator_g.evaluate_spline(x, a_g, b_g, c_g, d_g, flag=1) for x in x_test])
sp = np.array([interpolator_piecewise.evaluate_spline(x, a_p, b_p, c_p, d_p, flag=1) for x in x_test])

# Evaluate the exact functions at test points
exact_g = g(x_test)
exact_piece = np.array([piecewise_cubic(x, x_mesh) for x in x_test])

# Calculate errors
error_g = np.abs(gp - exact_g)
error_piece = np.abs(sp - exact_piece)
max_error_g = np.max(error_g)
max_error_piece = np.max(error_piece)

# Print the maximum errors
print(f"Max Error for cubic polynomial test: {max_error_g:.2e}")
print(f"Max Error for piecewise cubic test: {max_error_piece:.2e}")

# Plot the results
plt.figure(figsize=(15, 10))

# Plot for the cubic polynomial g(x)
plt.subplot(2, 2, 1)
plt.plot(x_test, exact_g, label='Ground Truth g(x)', color='blue')
plt.plot(x_test, gp, label='Spline Approximation', linestyle='dashed', color='orange')
plt.scatter(x_mesh, y_values_g, label='Mesh Points', color='black')
plt.title('Cubic Polynomial Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot for the piecewise cubic
plt.subplot(2, 2, 2)
plt.plot(x_test, exact_piece, label='Ground Truth Piecewise', color='green')
plt.plot(x_test, sp, label='Spline Approximation', linestyle='dashed', color='red')
plt.scatter(x_mesh, y_values_piece, s=30, label='Mesh Points', color='black')
plt.title('Piecewise Cubic Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot the errors
plt.subplot(2, 2, 3)
plt.semilogy(x_test, error_g + 1e-16)  # Add small value to handle zeros
plt.title(f'Error for Cubic Polynomial (Max: {max_error_g:.2e})')
plt.xlabel('x')
plt.ylabel('Error (log scale)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.semilogy(x_test, error_piece + 1e-16)  # Add small value to handle zeros
plt.title(f'Error for Piecewise Cubic (Max: {max_error_piece:.2e})')
plt.xlabel('x')
plt.ylabel('Error (log scale)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cubic_spline_verification.png')
plt.show()

# Additional verification
print("\nVerification Summary:")
print(f"- Cubic polynomial g(x) = 2x³ - 3x² + x + 4")
print(f"- Error should be close to machine precision: {max_error_g:.2e}")
print(f"- Piecewise cubic function with different cubics on each interval")
print(f"- Error should be close to machine precision: {max_error_piece:.2e}")

# Check if test passes (tolerance based on floating-point precision)
tolerance = 1e-10
if max_error_g < tolerance and max_error_piece < tolerance:
    print("\nTEST PASSED: Cubic spline correctly reproduces both functions!")
else:
    print("\nTEST FAILED: Errors exceed expected tolerance.")
    if max_error_g >= tolerance:
        print(f"- Cubic polynomial test failed: error = {max_error_g:.2e}")
    if max_error_piece >= tolerance:
        print(f"- Piecewise cubic test failed: error = {max_error_piece:.2e}")
