import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import seaborn as sns

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
    # {
    #     'name': 'Quartic Function',
    #     'function': lambda x: x ** 4 - 3 * x ** 2 + 2,
    #     'interval': (-1, 1),
    #     'first_max_deriv': 7,
    #     'second_max_deriv': 12,
    #     'fourth_max_deriv': 24,
    #     'true_value': 12 / 5
    # }
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

results_list = []
#m_values = [1 + 2 * i for i in range(0, 15)]
epsilon_values = [1e-3, 1e-4, 1e-5, 1e-7, 1e-8, 1e-9, 1e-10]


for f_data in functions:
    name = f_data['name']
    f = f_data['function']
    f_counter = FunctionWithCounter(f)
    a, b = f_data['interval']
    f1_max = f_data['first_max_deriv']
    f2_max = f_data['second_max_deriv']
    f4_max = f_data['fourth_max_deriv']
    y_true = f_data['true_value']
    m=1

    print(f'\n--- {name} ---')

    for epsilon in epsilon_values:

        print(f'\n--- {epsilon} ---')


        print(f'Composite Midpoint Rule')

        quad_midpoint = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f2_max)
        result = quad_midpoint.composite_midpoint(optimal_H_m=True)
        # print(f'\nThe integral value from Composite Midpoint Rule is: {result}')
        # print(f'The optimal H_m and m are: H_m = {quad_midpoint.H_m}, m = {quad_midpoint.m}')
        error_estimate = ((b-a)/24) * quad_midpoint.H_m**2 * f2_max
        # print(f'Estimated error is: {error_estimate}')
        error = np.abs(y_true-result)
        print(f'Resulting Error is: {error}')
        print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Midpoint Rule',
            'adaptive': False,
            'm': quad_midpoint.m,
            'H_m': quad_midpoint.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })



        print(f'\nComposite Trapezoidal Rule')

        f_counter.counter = 0
        quad_trapezoid = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f2_max)
        result = quad_trapezoid.composite_trapezoid(optimal_H_m=True)
        # print(f'\nThe integral value from Composite Trapezoid Rule is: {result}')
        # print(f'The optimal H_m and m are: H_m = {quad_trapezoid.H_m}, m = {quad_trapezoid.m}')
        error_estimate = ((b - a) / 12) * quad_trapezoid.H_m ** 2 * f2_max
        # print(f'Estimated Error: {error_estimate}')
        error = np.abs(y_true - result)
        print(f'Resulting Error is: {error}')
        print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Trapezoidal Rule',
            'adaptive': False,
            'm': quad_trapezoid.m,
            'H_m': quad_trapezoid.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })


        # print(f'\nComposite 2-Point Rule')

        f_counter.counter = 0
        quad_2_point = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f2_max)
        result = quad_2_point.composite_2_point(optimal_H_m=True)
        # print(f'\nThe integral value from Composite 2-Point Rule is: {result}')
        # print(f'The optimal H_m and m are: H_m = {quad_2_point.H_m}, m = {quad_2_point.m}')
        error_estimate = ((b - a) / 36) * quad_2_point.H_m ** 2 * f2_max
        # print(f'Estimated Error: {error_estimate}')
        error = np.abs(y_true - result)
        # print(f'Resulting Error is: {error}')
        # print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite 2-Point Rule',
            'adaptive': False,
            'm': quad_2_point.m,
            'H_m': quad_2_point.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })

        # print(f'\nComposite Simpsons First Rule')

        f_counter.counter = 0
        quad_simpson_first = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f4_max)
        result = quad_simpson_first.composite_simpson_first(optimal_H_m=True)
        # print(f'\nThe integral value from Composite Simpsons First Rule is: {result}')
        # print(f'The optimal H_m and m are: H_m = {quad_simpson_first.H_m} and m = {quad_simpson_first.m}')
        error_estimate = ((b-a) / 2880) * quad_simpson_first.H_m**4 * f4_max
        # print(f'Estimated Error: {error_estimate}')
        error = np.abs(y_true - result)
        # print(f'Resulting Error is: {error}')
        # print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Simpsons First Rule',
            'adaptive': False,
            'm': quad_simpson_first.m,
            'H_m': quad_simpson_first.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })

        # print(f'\nComposite Gauss-Legendre')

        f_counter.counter = 0
        quad_gauss = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f4_max)
        result = quad_gauss.gauss_2_point_quadrature(optimal_H_m=True)
        # print(f'\nThe integral value from Composite Gauss-Legendre 2-Point Rule is: {result}')
        # print(f'The optimal H_m and m are: H_m = {quad_gauss.H_m}, m = {quad_gauss.m}')
        error_estimate = ((b-a) / 4320) * quad_gauss.H_m**4 * f4_max
        # print(f'Estimated Error: {error_estimate}')
        error = np.abs(y_true - result)
        # print(f'Resulting Error is: {error}')
        # print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Gauss-Legendre 2-Point Rule',
            'adaptive': False,
            'm': quad_gauss.m,
            'H_m': quad_gauss.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })


        if epsilon >= 1e-6:
            # print(f'\nComposite Left Rectangle')

            f_counter.counter = 0
            quad_left_rectangle = NumericalQuadrature(a, b, m, f_counter, epsilon=epsilon, f_deriv_max=f1_max)
            result = quad_left_rectangle.composite_left_rectangle(optimal_H_m=True)
            # print(f'\nThe integral value from Composite Left-Rectangle Rule is: {result}')
            # print(f'The H_m and m being used are: H_m = {quad_left_rectangle.H_m}, m = {quad_left_rectangle.m}')
            error_estimate = ((b - a) / 2) * quad_left_rectangle.H_m * f1_max
            # print(f'Estimated Error: {error_estimate}')
            error = np.abs(y_true - result)
            # print(f'Resulting Error is: {error}')
            # print(f'Number of Function Evaluations: {f_counter.counter}')

            results_list.append({
                'function': name,
                'Method': 'Composite Left Rectangle Rule',
                'adaptive': False,
                'm': quad_left_rectangle.m,
                'H_m': quad_left_rectangle.H_m,
                'True Error': error,
                'Error Estimate': error_estimate,
                'func_evals': f_counter.counter,
                'epsilon': epsilon
            })
        else:
            print(f'Skipping Composite Left Rectangle Rule for epsilon = {epsilon} (too small for reliability)')

        print(f'\nAdaptive Refinement for Midpoint Rule')

        f_counter.counter = 0
        quad_mid_adapt = NumericalQuadrature(a, b, m, f_counter)
        result = quad_mid_adapt.composite_midpoint(optimal_H_m=False, adaptive=True, tol=epsilon, max_iter=2000,
                                                   y_true=y_true)
        # print(f'\nAdaptive Refinement result for Composite Midpoint is: {result}')
        # print(f'Adaptive Refinement H_m and m are: H_m = {quad_mid_adapt.H_m}, m = {int(np.ceil(quad_mid_adapt.m))}')
        error_estimate = ((b - a) / 24) * quad_mid_adapt.H_m ** 2 * f2_max
        # print(f'Estimated error is: {error_estimate}')
        error = np.abs(y_true - result)
        print(f'Adaptive Refinement error is: {error}')
        print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Midpoint Rule',
            'adaptive': True,
            'm': quad_mid_adapt.m,
            'H_m': quad_mid_adapt.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })

        print(f'\nAdaptive Refinement for Trapezoidal Rule')

        f_counter.counter = 0
        quad_trap_adapt = NumericalQuadrature(a, b, m, f_counter)
        result = quad_trap_adapt.composite_trapezoid(optimal_H_m=False, adaptive=True, tol=epsilon, max_iter=2000,
                                                     y_true=y_true)
        # print(f'\nAdaptive Refinement result for Composite Trapezoidal is: {result}')
        # print(f'Adaptive Refinement H_m and m are: H_m = {quad_trap_adapt.H_m}, m = {int(np.ceil(quad_trap_adapt.m))}')
        error_estimate = ((b - a) / 12) * quad_trap_adapt.H_m ** 2 * f2_max
        # print(f'Estimated Error: {error_estimate}')
        error = np.abs(y_true - result)
        print(f'Adaptive Refinement error is: {error}')
        print(f'Number of Function Evaluations: {f_counter.counter}')

        results_list.append({
            'function': name,
            'Method': 'Composite Trapezoidal Rule',
            'adaptive': True,
            'm': quad_trap_adapt.m,
            'H_m': quad_trap_adapt.H_m,
            'True Error': error,
            'Error Estimate': error_estimate,
            'func_evals': f_counter.counter,
            'epsilon': epsilon
        })


# Setting results into a full dataframe to extract different parts later
df = pd.DataFrame(results_list)

# Filtering the dataframe
df_func = df[~df['adaptive']]
df_func = df_func[df_func['function'] == 'Function 2']

# Creating the color palette to be consistent across all
palette = {
    'Composite Trapezoidal Rule': '#1f77b4',            # Blue
    'Composite Midpoint Rule': '#ff7f0e',               # Orange
    'Composite Gauss-Legendre 2-Point Rule': '#2ca02c', # Green
    'Composite Simpsons First Rule': '#d62728',         # Red
    'Composite 2-Point Rule': '#9467bd',                # Purple
    'Composite Left Rectangle Rule': '#8c564b'          # Brown
}

# Melting the dataframe for ease of use for only the values needed for this plot
df_melted_m = df_func.melt(
    id_vars=['m', 'Method'],
    value_vars=['True Error', 'Error Estimate'],
    var_name='Error Type',
    value_name='error'
)
df_melted_m['Method'] = df_melted_m['Method'].str.title()
df_melted_m['Error Type'] = df_melted_m['Error Type'].str.replace('_', ' ').str.title()

# Setting the font properties of the legend
bold_font = FontProperties()
bold_font.set_weight('bold')
bold_font.set_size('medium')

# Figure of Number of Subintervals vs Error
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted_m, x='m', y='error', hue='Method', palette=palette, style='Error Type', markers=True)
plt.xscale('log')
plt.yscale('log')
plt.title(f'Error vs Number of Subintervals for Integral 2')
plt.xlabel('Number of Subintervals (m) (log scale)')
plt.ylabel('Error (log scale)')
plt.legend(bbox_to_anchor=(1.05,1), title='Method and Error Type', title_fontproperties=bold_font)
plt.tight_layout()
plt.show()

# Create a different melted dataframe for the next plot
df_melted_eps = df_func.melt(
    id_vars=['epsilon', 'Method'],
    value_vars=['True Error', 'Error Estimate'],
    var_name='Error Type',
    value_name='error'
)
df_melted_eps['Method'] = df_melted_eps['Method'].str.title()
df_melted_eps['Error Type'] = df_melted_eps['Error Type'].str.replace('_', ' ').str.title()

# Figure of Required Accuracy vs Accuracy Achieved
plt.figure(figsize=(10,6))
sns.lineplot(data=df_melted_eps, x='epsilon', y='error', hue='Method', palette=palette, style='Error Type', markers=True, dashes=False)
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'$\varepsilon$ (log scale)')
plt.ylabel('Error (log scale)')
plt.legend(bbox_to_anchor=(1.05,1), title='Method and Error Type', title_fontproperties=bold_font)
plt.title(r'Error vs Required Accuracy ($\varepsilon$) for Integral 2')
plt.tight_layout()
plt.show()


# Setting up Efficieny and Complexity plots
df_func = df_func.sort_values(by='epsilon', ascending=True)

# Initializing the plot
fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True)
fig.suptitle(r'Efficiency and Complexity for Integral 2: $\quad \int_{0}^{\frac{\pi}{3}} e^{\sin(2x)}\cos(2x) \, dx$')

# Function Evaluations vs Epsilon
sns.lineplot(data=df_func, x='epsilon', y='func_evals', hue='Method', palette=palette, marker='o', ax=axes[0])
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title(r'Function Evaluations vs Required Accuracy ($\varepsilon$)')
axes[0].set_xlabel(r'$\varepsilon$ (log scale)')
axes[0].set_ylabel('Function Evaluations')
axes[0].get_legend().remove()

# Number of Subintervals vs Epsilon
sns.lineplot(data=df_func, x='epsilon', y='m', hue='Method', palette=palette, marker='o', ax=axes[1])
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title(r'Number of Subintervals (m) vs Required Accuracy ($\varepsilon$)')
axes[1].set_xlabel(r'$\varepsilon$ (log scale)')
axes[1].set_ylabel('Number of Subintervals (m) (log scale)')
axes[1].get_legend().remove()

# Getting the labels and their respective values for the legend creation
handles, labels = axes[0].get_legend_handles_labels()

# Creating the legend
fig.legend(handles, labels, bbox_to_anchor=(0.75,1), loc='upper left', title='Method', title_fontproperties=bold_font, frameon=True)

# Invert so that we go from largest to smallest epsilon
plt.gca().invert_xaxis()
plt.tight_layout(rect=[0,0,0.75,1])
plt.show()

### ADAPTIVE ###
adapt_methods = ['Composite Midpoint Rule', 'Composite Trapezoidal Rule']
df_func = df[df['function'] == 'Function 5']
df_func = df_func[df_func['Method'].isin(adapt_methods)]
df_func['accuracy_per_eval'] = 1 / (df_func['True Error'] * df_func['func_evals'])

plt.figure(figsize=(10,6))
sns.lineplot(data=df_func, x='epsilon', y='accuracy_per_eval', hue='Method', style='adaptive', markers=True, dashes=False)
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel(r'$\varepsilon$ (log scale)')
plt.ylabel('Accuracy Per Function Evaluation')
plt.title('Efficiency: Adaptive vs A Priori Refinement For Integral 5')
plt.legend(title='Method and Adaptive', title_fontproperties=bold_font)
plt.tight_layout()
plt.show()





