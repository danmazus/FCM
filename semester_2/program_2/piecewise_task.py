import numpy as np
import matplotlib.pyplot as plt
from semester_2.interpolation_class import PolynomialInterpolation

# ============= Functions ================
def f(x):
    #return np.sin(x)
    #return 3 * np.power(x, 3) + 4 * np.power(x, 2) + 2 * x + 1
    return 4 * np.power(x, 5) + (1 / 2) * np.power(x, 2) + 3 * x


def df(x):
    #return np.cos(x)
    #return 9 * np.power(x, 2) + 8 * x + 2
    return 20 * np.power(x, 4) + x + 3

function_description_for_title = {
    "f_1": r'$\sin(x)$',
    "f_2": r'$3x^3 + 4x^2 + 2x + 1$',
    "f_3": r'$4x^5 + \frac{1}{2}x^2 + 3x$'
}

# ============= Settings ================

# Bounds
#a, b = 0, 4 * np.pi
a, b = -4, 4

# Number of points to evaluate at
n = 100

# Number of subintervals
M = 5

# Degrees to be Tested (1 - Linear, 2 - Quadratic, 3 - Cubic)
degrees = [1, 2, 3]

# Mesh Types to test (1 - Uniform, 3 - Chebyshev Points of the Second Kind)
meshes = [1, 3]

# Precision
dtype = np.float64

# x-values to estimate
x_points = np.linspace(a, b, n)

# x-values to graph f(x)
x_eval = np.linspace(a, b, 1000)

# Associated y-values for each
exact_value = f(x_points)
exact_plot = f(x_eval)

# ================ Piecewise Interpolation ======================

# Loop over each type of mesh
for flag in meshes:

    # Initialize the figure for each mesh type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting f(x)
    ax = axes[0]
    ax.plot(x_eval, exact_plot, 'k', label='f(x)')
    ax.set_title(f'Function, {function_description_for_title["f_3"]}, and Interpolations (Flag={flag})')

    # Initialize dictionary to store the errors
    errors = {}

    # Loop over each of the degrees
    for d in degrees:
        # Initialize the Interpolator
        interpolator = PolynomialInterpolation(a, b, d, dtype, M)

        # Create the Mesh
        interpolator.local_mesh(flag)
        interpolator.mesh_points(flag, piecewise=True)

        # Compute the divided difference coefficients
        interpolator.newton_divdiff(f, piecewise=True)

        # Compute the interpolated values for each x_point
        interp_values = np.zeros_like(x_points)
        for i, x in enumerate(x_points):
            interp_values[i] = interpolator.piecewise_interpolation(x, f, df, flag, hermite=False)

        # Append the error for the given degree
        errors[d] = np.abs(interp_values - exact_value)

        # Plot the interpolated polynomial for the degree
        ax.plot(x_points, interp_values, '--', label=f'd={d}')

    # ============= Hermite Cubic ==================

    # Initialize the Interpolator again for new interpolation
    interpolator = PolynomialInterpolation(a, b, 3, dtype, M)

    # Initiate the mesh
    interpolator.local_mesh(flag)
    interpolator.mesh_points(flag, piecewise=True)

    # Compute the divided difference coefficients
    interpolator.newton_divdiff(f, piecewise=True)

    interp_values = np.zeros_like(x_points)

    # Compute the interpolated values
    for i, x in enumerate(x_points):
        interp_values[i] = interpolator.piecewise_interpolation(x, f, df, flag, hermite=True)

    # Append the dictionary for the hermite errors
    errors['Hermite'] = np.abs(interp_values - exact_value)

    # Plot the Hermite Cubic Interpolated Polynomial
    ax.plot(x_points, interp_values, '--', label='Hermite Cubic')

    ax.legend(loc='best', fontsize='x-small')
    ax.grid(True)

    # Error Plot
    ax_err = axes[1]
    for key, err in errors.items():
        ax_err.semilogy(x_points, err, label=f'Error (d={key})')
    ax_err.set_title(f'Interpolation Error (Flag={flag})')
    ax_err.legend(loc = 'upper left', fontsize='x-small')
    ax_err.grid(True)

    plt.tight_layout()
    plt.show()