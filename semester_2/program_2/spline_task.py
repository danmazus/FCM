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

# Bounds
#a, b = 0, 4 * np.pi
a, b = -4, 4

dff_a, dff_b = df2(a), df2(b)

# Number of points to evaluate at
n = 100

# Number of Intervals for Spline (d + 1) Mesh points
d = [2, 3, 5, 8, 12]

# Configuration for spline interpolation
piecewise = False

# Mesh Types to test (1 - Uniform, 3 - Chebyshev Points of the Second Kind)
meshes = [1, 3]

# Precision
dtype = np.float64

# x-values to estimate
x_points = np.linspace(a, b, n)

# x-values to graph f(x)
x_eval = np.linspace(a, b, 1000)

# Exact value of points
exact_value = f(x_points)
exact_plot = f(x_eval)

# ===================== Spline Interpolation =====================
for flag in meshes:
    # Create a new figure for each mesh type (flag)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting f(x) on axes[0]
    ax = axes[0]
    ax.plot(x_eval, exact_plot, 'k', label='f(x)')

    # Initialize dictionary to store the errors
    errors = {}

    for d_values in d:
        spline_interpolator = PolynomialInterpolation(a, b, d_values, dtype)
        spline_interpolator.mesh_points(flag, piecewise)

        second_derivatives = spline_interpolator.spline_interpolation(f, flag, second_deriv_0=dff_a,
                                                                      second_deriv_n=dff_b)
        sp_a, sp_b, sp_c, sp_d = spline_interpolator.get_spline_coefficients(second_derivatives, f, flag)

        sp_interp_values = np.zeros_like(x_points)
        for i, x in enumerate(x_points):
            sp_interp_values[i] = spline_interpolator.evaluate_spline(x, sp_a, sp_b, sp_c, sp_d, flag)

        errors[d_values] = np.abs(sp_interp_values - exact_value)

        # Plot the spline interpolation result for this mesh type and d value on the same plot
        ax.plot(x_points, sp_interp_values, '--', label=f'd={d_values}')

    # Customize the first subplot
    ax.legend(loc='best', fontsize='x-small')
    ax.set_title(f'Function, $f(x) =${function_description_for_title["f_3"]}, and Spline Interpolations (Flag={flag})')
    ax.grid(True)

    # Plotting the errors on axes[1]
    ax_err = axes[1]
    for d_values, err in errors.items():
        ax_err.semilogy(x_points, err, label=f'Flag: {flag}, d={d_values}')

    ax_err.set_title(f'Interpolation Errors (Flag={flag}), $f(x) = ${function_description_for_title["f_3"]}')
    ax_err.legend(loc='best', fontsize='x-small')
    ax_err.grid(True)

    # Adjust layout and show the plot for this flag
    plt.tight_layout()
    plt.show()