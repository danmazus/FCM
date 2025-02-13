# Defining Functions
def p_1(d, rho):
    return lambda x: (x - rho) ** d

def p_2(d):
    for i in range(d):
        prod = 1.0
        return lambda x: (x - i)

def p_3(n):
    return lambda x: (x - 1)

def p_4(n):
    return lambda x: 1 / (1 + (25 * x)^n)
