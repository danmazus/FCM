# Defining Functions
def f_1(d, rho):
    return lambda x: (x - rho) ** d

def f_2(d):
    for i in range(d):
        return lambda x: (x - i)

def f_3(n):
    return lambda x: (x - 1)

def f_4(n):
    return lambda x: 1 / (1 + (25 * x)^n)
