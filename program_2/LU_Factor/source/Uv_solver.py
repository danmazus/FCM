def solve_Ux(LU, y, n):
    """Solves the Equation Ux = y with Backwards Substitution

    Parameters include:
    LU: a combination matrix of unit lower and upper triangular matrices stored in single 2D array
    y: the output vector from Lb_solver
    n: the dimension specified by user.

    Returns:
    x: the solution vector.
    """
    # Initialize x
    x = [0 for i in range(n)]

    # Starting from last row going upwards (Backward Substitution)
    for i in range(n-1, -1, -1):
        temp_sum = 0

        # Calculate the sums for k > i
        for k in range(i + 1, n):
            temp_sum += LU[i][k] * x[k]

        # Solving for x[i]
        x[i] = (y[i] - temp_sum)/LU[i][i]

    return x