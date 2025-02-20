import numpy as np

# Defining Functions
def p_1(d, rho):
    return lambda x: (x - rho) ** d

def p_2(d):
    return lambda x: np.prod(x - np.arange(1, d+1))

def p_3(n):
    return lambda x: (x - 1)

def p_4(n):
    return lambda x: 1 / (1 + (25 * x) ** n)

d = 5
f = p_2(d)
print(f)
x = 10
print(f(x))
