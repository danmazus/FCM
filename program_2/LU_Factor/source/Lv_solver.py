def solve_Lb(LU, b, n):
    """Solves the equation Ly = b (forward substitution)

    Parameters include:
    LU: a combination matrix of unit lower and upper triangular matrices stored in single 2D array
    b: the vector from the equation Ax = b
    n: the dimension of vector and matrix given by user

    Returns:
    y: the vector to use in Ux solver (Ux = y)
    """
    # Initializing vector y
    y = [0 for i in range(n)]

    # Setting the first element
    y[0] = b[0]

    for i in range(n):
        # Initializing temp_sum variable
        temp_sum = 0

        # Summing the L[i][k] * y[k]
        for k in range(i):
            # Accumulating the sum before updating
            temp_sum += LU[i][k] * y[k]
        y[i] = b[i] - temp_sum

    return y

