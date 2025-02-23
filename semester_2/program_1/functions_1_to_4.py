import numpy as np
from numpy.core.numeric import newaxis


# Defining Functions
def p_1(d, rho):
    return lambda x: (x - rho) ** d

def p_2(d):
    return lambda x: np.prod(x[:, None] - np.arange(1, d+1), axis=1)

def p_3(x_mesh, x_values):
    y = np.zeros(len(x_values))
    for j in range(len(x_values)):
        if np.isclose(x_values[j], x_mesh, atol=1e-16).any() and x_values[j] != x_mesh[-1]:
            y[j] = 0
        elif x_values[j] == x_mesh[-1]:
            y[j] = 1
        else:
            temp_prod = 1
            for i in range(len(x_mesh) - 1):
                temp_prod *= (x_values[j] - x_mesh[i]) / (x_mesh[-1] - x_mesh[i])

            y[j] = temp_prod

    return y

def p_4(n):
    return lambda x: 1 / (1 + (25 * x) ** n)

