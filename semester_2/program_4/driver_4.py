import numpy as np
import matplotlib.pyplot as plt
from semester_2.quadrature_funcs import NumericalQuadrature

class FunctionWithCounter:
    def __init__(self, f):
        self.f = f
        self.counter = 0

    def __call__(self, x):
        self.counter += np.size(x)
        return self.f(x)

    def reset_counter(self):
        self.counter = 0

functions = [
    # {
    #     'name': 'Linear Function',
    #     'function': lambda x: 2 * x + 1,
    #     'interval': (0, 3),
    #     'first_max_deriv': 2,
    #     'second_max_deriv': 0,
    #     'fourth_max_deriv': 0,
    #     'true_value': 12
    # },
    # {
    #     'name': 'Quadratic Function',
    #     'function': lambda x: x ** 2 + 3 * x - (1 / 2),
    #     'interval': (-1, 2),
    #     'first_max_deriv': 7,
    #     'second_max_deriv': 2,
    #     'fourth_max_deriv': 0,
    #     'true_value': 6
    # },
    # {
    #     'name': 'Cubic Function',
    #     'function': lambda x: 3 * x ** 3 + 2 * x ** 2 + (1/2) * x + 3,
    #     'interval': (-3, 4),
    #     'first_max_deriv': 160.5,
    #     'second_max_deriv': 76,
    #     'fourth_max_deriv': 0,
    #     'true_value': 644/3
    # },
    {
        'name': 'Function 1',
        'function': lambda x: np.exp(x),
        'interval': (0, 3),
        'first_max_deriv': np.exp(3),
        'second_max_deriv': np.exp(3),
        'fourth_max_deriv': np.exp(3),
        'true_value': np.exp(3) - 1
    },
    {
        'name': 'Function 2',
        'function': lambda x: np.exp(np.sin(2 * x)) * np.cos(2 * x),
        'interval': (0, np.pi/3),
        'first_max_deriv': 5.43656,
        'second_max_deriv': 16.28308,
        'fourth_max_deriv': 396.97529,
        'true_value': (1 / 2) * (-1 + np.exp(np.sqrt(3) / 2))
    },
    {
        'name': 'Function 3',
        'function': lambda x: np.tanh(x),
        'interval': (-2, 1),
        'first_max_deriv': 1,
        'second_max_deriv': 0.7698,
        'fourth_max_deriv': 4.0859,
        'true_value': np.log((np.cosh(1) / np.cosh(2)))
    },
    {
        'name': 'Function 4',
        'function': lambda x: x * np.cos(2 * np.pi * x),
        'interval': (0, 3.5),
        'first_max_deriv': 20.51775,
        'second_max_deriv': 138.17466,
        'fourth_max_deriv': 5454.9091,
        'true_value': (-1 / (2 * np.power(np.pi, 2)))
    },
    {
        'name': 'Function 5',
        'function': lambda x: x + (1 / x),
        'interval': (0.1, 2.5),
        'first_max_deriv': 99,
        'second_max_deriv': 2000,
        'fourth_max_deriv': 2400000,
        'true_value': ((2.5 ** 2 - 0.1 ** 2) / 2) + np.log(2.5 / 0.1)
    },
]

for f_data in functions:
    name = f_data['name']
    f = f_data['function']
    f_counter = FunctionWithCounter(f)
    a, b = f_data['interval']
    f1_max = f_data['first_max_deriv']
    f2_max = f_data['second_max_deriv']
    f4_max = f_data['fourth_max_deriv']
    y_true = f_data['true_value']
    m=10
    epsilon = 1e-8

    print(f'\n--- {name} ---')


    print(f'Composite Midpoint Rule')

    quad_midpoint = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f2_max)
    result = quad_midpoint.composite_midpoint(optimal_H_m=True)
    print(f'\nThe integral value from Composite Midpoint Rule is: {result}')
    print(f'The optimal H_m and m are: H_m = {quad_midpoint.H_m}, m = {quad_midpoint.m}')
    error_estimate = ((b-a)/24) * quad_midpoint.H_m**2 * f2_max
    print(f'Estimated error is: {error_estimate}')
    error = np.abs(y_true-result)
    print(f'Resulting Error is: {error}')
    print(f'Number of Function Evaluations: {f_counter.counter}')

    print(f'\nAdaptive Refinement for Midpoint Rule')

    f_counter.counter = 0
    quad_mid_adapt = NumericalQuadrature(a, b, m, f_counter, f_deriv_max=f2_max)
    result = quad_mid_adapt.composite_midpoint(optimal_H_m=False, adaptive=True, tol=epsilon, max_iter=2000, y_true=y_true)
    print(f'\nAdaptive Refinement result for Composite Midpoint is: {result}')
    print(f'Adaptive Refinement H_m and m are: H_m = {quad_mid_adapt.H_m}, m = {int(np.ceil(quad_mid_adapt.m))}')
    error_estimate = ((b-a)/24) * quad_mid_adapt.H_m**2 * f2_max
    print(f'Estimated error is: {error_estimate}')
    error = np.abs(y_true-result)
    print(f'Adaptive Refinement error is: {error}')
    print(f'Number of Function Evaluations: {f_counter.counter}')


    print(f'\nComposite Trapezoidal Rule')

    f_counter.counter = 0
    quad_trapezoid = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f2_max)
    result = quad_trapezoid.composite_trapezoid(optimal_H_m=True)
    print(f'\nThe integral value from Composite Trapezoid Rule is: {result}')
    print(f'The optimal H_m and m are: H_m = {quad_trapezoid.H_m}, m = {quad_trapezoid.m}')
    error_estimate = ((b - a) / 12) * quad_trapezoid.H_m ** 2 * f2_max
    print(f'Estimated Error: {error_estimate}')
    error = np.abs(y_true - result)
    print(f'Resulting Error is: {error}')
    print(f'Number of Function Evaluations: {f_counter.counter}')

    print(f'\nAdaptive Refinement for Trapezoidal Rule')

    f_counter.counter = 0
    quad_trap_adapt = NumericalQuadrature(a, b, m, f_counter, f_deriv_max=f2_max)
    result = quad_trap_adapt.composite_trapezoid(optimal_H_m=False, adaptive=True, tol=epsilon, max_iter=2000,
                                                 y_true=y_true)
    print(f'\nAdaptive Refinement result for Composite Trapezoidal is: {result}')
    print(f'Adaptive Refinement H_m and m are: H_m = {quad_trap_adapt.H_m}, m = {int(np.ceil(quad_trap_adapt.m))}')
    error_estimate = ((b-a) / 12) * quad_trap_adapt.H_m ** 2 * f2_max
    print(f'Estimated Error: {error_estimate}')
    error = np.abs(y_true - result)
    print(f'Adaptive Refinement error is: {error}')
    print(f'Number of Function Evaluations: {f_counter.counter}')


    print(f'\nComposite 2-Point Rule')

    quad_2_point = NumericalQuadrature(a, b, m, f, epsilon=epsilon, f_deriv_max=f2_max)
    result = quad_2_point.composite_2_point(optimal_H_m=True)
    print(f'\nThe integral value from Composite 2-Point Rule is: {result}')
    print(f'The optimal H_m and m are: H_m = {quad_2_point.H_m}, m = {quad_2_point.m}')
    error_estimate = ((b - a) / 36) * quad_2_point.H_m ** 2 * f2_max
    print(f'Estimated Error: {error_estimate}')
    error = np.abs(y_true - result)
    print(f'Resulting Error is: {error}')

    print(f'\nComposite Simpsons First Rule')

    quad_simpson_first = NumericalQuadrature(a, b, m, f, epsilon=epsilon, f_deriv_max=f4_max)
    result = quad_simpson_first.composite_simpson_first(optimal_H_m=True)
    print(f'\nThe integral value from Composite Simpsons First Rule is: {result}')
    print(f'The optimal H_m and m are: H_m = {quad_simpson_first.H_m} and m = {quad_simpson_first.m}')
    error_estimate = ((b-a) / 2880) * quad_simpson_first.H_m**4 * f4_max
    print(f'Estimated Error: {error_estimate}')
    error = np.abs(y_true - result)
    print(f'Resulting Error is: {error}')

    print(f'\nComposite Gauss-Legendre')

    quad_gauss = NumericalQuadrature(a, b, m, f, epsilon=epsilon, f_deriv_max=f4_max)
    result = quad_gauss.gauss_2_point_quadrature(optimal_H_m=True)
    print(f'\nThe integral value from Composite Gauss-Legendre 2-Point Rule is: {result}')
    print(f'The optimal H_m and m are: H_m = {quad_gauss.H_m}, m = {quad_gauss.m}')
    error_estimate = ((b-a) / 4320) * quad_gauss.H_m**4 * f4_max
    print(f'Estimated Error: {error_estimate}')
    error = np.abs(y_true - result)
    print(f'Resulting Error is: {error}')


# # quad_left_rectangle = NumericalQuadrature(a, b, m, f, epsilon=epsilon, f_deriv_max=f1_max)
# # result = quad_left_rectangle.composite_left_rectangle(optimal_H_m=True)
# # print(f'\nThe integral value from Composite Left-Rectangle Rule is: {result}')
# # print(f'The H_m and m being used are: H_m = {quad_left_rectangle.H_m}, m = {quad_left_rectangle.m}')
# # error_estimate = ((b - a) / 2) * quad_left_rectangle.H_m * f1_max
# # print(f'Estimated Error: {error_estimate}')
# # error = np.abs(y_true - result)
# # print(f'Resulting Error is: {error}')


