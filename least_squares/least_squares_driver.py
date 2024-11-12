import numpy as np
from my_package.functions_storage import (
    compute_house_np,
    vec_2_norm_np,
    frob_norm_np,
    generate_unit_upper_triangular_np,
    generate_nonsingular_upper_triangular_np,
    generate_float_1D_vector_np
)

# User Inputs for Solver
def get_user_inputs():
    """Get problem parameters from user input.

    Prompts user for:
    - Matrix dimensions (n rows, k columns)
    - Number of random reflectors
    - Problem type (simple test, uniform random, normal random)
    - Range for random reflectors (dmin, dmax)
    - Debug output preference
    """
    print("\nLeast Squares Solver Configuration")
    print("----------------------------------")

    while True:
        try:
            n = int(input("Enter number of rows (n) [default=10]: ") or "10")
            k = int(input("Enter number of columns (k) [default=3]: ") or "3")
            if k > n:
                print("Error: k must be less than or equal to n")
                continue
            s = int(input("Enter number of random reflectors (s) [default=10]: ") or "10")

            print("\nChoose problem type:")
            print("1. Simple test case (ones and eye matrix)")
            print("2. Uniform random numbers")
            print("3. Normal random numbers")
            problem_type = int(input("Enter problem type (1-3) [default=1]: ") or "1")

            if problem_type not in [1, 2, 3]:
                print("Error: Invalid problem type")
                continue

            # New inputs for random reflector range
            print("\nSet range for random reflector values:")
            dmin = float(input("Enter minimum value [default=-10.0]: ") or "-10.0")
            dmax = float(input("Enter maximum value [default=10.0]: ") or "10.0")
            if dmin >= dmax:
                print("Error: Minimum value must be less than maximum value")
                continue

            # New input for ratio factor (to scale the random matrices)
            ratio_factor = float(input("Enter ratio factor for random matrix scaling [default=0.5]: ") or "0.5")

            debug = input("Enable debug output? (y/n) [default=n]: ").lower().startswith('y')

            return n, k, s, problem_type, dmin, dmax, ratio_factor, debug

        except ValueError:
            print("Error: Please enter valid numbers")

# Least Squares Solver with
def least_squares_solver_interactive():
    """Interactive least squares solver using custom matrix operations"""
    n, k, s, problem_type, dmin, dmax, ratio_factor, debug = get_user_inputs()

    # Setting Seed for Reproducibility: Comment out if wanting to test different
    np.random.seed(42)

    # Initialize problem matrices using custom generation functions
    if problem_type == 1:
        # Simple test case using unit upper triangular, that is k x k, with modifications
        R = generate_unit_upper_triangular_np(1, 10, k)
        for i in range(k):
            R[i, i] = 10.0  # Strengthen diagonal
        c = np.ones(k)

    elif problem_type == 2:
        # Uniform random case using nonsingular upper triangular, that is 'k x k' and vector that has 'k' elements
        R = generate_nonsingular_upper_triangular_np(dmin, dmax, k, ratio_factor)
        c = generate_float_1D_vector_np(dmin, dmax, k)

    else: # problem_type == 3
        # Normal random case using nonsingular upper triangular with wider range, that is 'k x k'
        R = generate_nonsingular_upper_triangular_np(dmin, dmax, k, ratio_factor)
        c = generate_float_1D_vector_np(dmin, dmax, k)

    # Normalize vectors using custom norm function
    c = c / vec_2_norm_np(c)
    d = np.ones(n - k)
    cdratio = 1.0
    d = (d / vec_2_norm_np(d)) * cdratio

    # Compute true solution using custom matrix operations
    x_true = np.linalg.solve(R, c)  # Keep this as is since it's initialization

    # Construct full right-hand side vectors
    b1_true = np.concatenate([c, np.zeros(n - k)])
    b2_true = np.concatenate([np.zeros(k), d])

    # Construct full system matrix
    A = np.vstack([R, np.zeros((n - k, k))])

    # Apply random reflectors using custom operations and NumPy Functions
    for j in range(s):
        # Generate random reflection vector that has 'n' elements
        v = generate_float_1D_vector_np(dmin, dmax, n)
        v = v / vec_2_norm_np(v)

        # Apply reflection to right-hand side vectors using custom multiply
        b1_true = b1_true - 2.0 * (np.dot(v, b1_true)) * v
        b2_true = b2_true - 2.0 * (np.dot(v, b2_true)) * v

        # Apply reflection to system matrix
        z = np.dot(v, A)  # Keep basic dot product for initialization
        A = A - 2.0 * np.outer(v, z)  # Keep outer product for initialization

    b = b1_true + b2_true
    Asave, bsave = A.copy(), b.copy()

    print("\nSolving least squares problem...")

    # Apply Householder transformations using custom operations and NumPy Functions
    for j in range(k - 1):
        nj = n - j      # Number of rows in active matrix
        nk = k - j - 1  # Number of columns that are updated by H_j which is one less than the active part of 'A'

        # Compute Householder vector using custom function
        u, rho = compute_house_np(A[j:n, j], nj)
        A[j, j] = rho
        A[j + 1:n, j] = 0

        # Apply Householder reflection using custom matrix operations
        v = np.dot(A[j:n, j + 1:k].T, u)  # Keep basic dot product for initialization
        A[j:n, j + 1:k] = A[j:n, j + 1:k] - 2.0 * np.outer(u, v)  # Keep outer product for initialization

        # Apply to right-hand side using custom operations
        housescalar = np.dot(u, b[j:n])  # Keep basic dot product for initialization
        b[j:n] = b[j:n] - 2.0 * housescalar * u

        if debug:
            print(f'\nA and b after H_{j}:')
            print('A =\n', A)
            print('b =\n', b)

    # Handle final column if needed
    if n > k:
        nj = n - k + 1
        u, rho = compute_house_np(A[k - 1:n, k - 1], nj)
        A[k - 1, k - 1] = rho
        A[k:n, k - 1] = 0
        housescalar = np.dot(u, b[k - 1:n])  # Keep basic dot product for initialization
        b[k - 1:n] = b[k - 1:n] - 2.0 * housescalar * u

        if debug:
            print(f'\nA and b after H_{k}:')
            print('A =\n', A)
            print('b =\n', b)

    # Extract and solve upper triangular system
    R_comp = A[:k, :k]
    c_comp = b[:k]
    x_comp = np.linalg.solve(R_comp, c_comp)  # Keep this as is since it's final solution

    # Compute components using custom matrix operations
    b1_comp = np.dot(Asave, x_comp)  # Keep basic dot product for final computation
    b2_comp = bsave - b1_comp

    # Compute errors using custom norm functions
    print("\nResults:")
    print("---------")
    errors = {
        'x error': vec_2_norm_np(x_comp - x_true),
        'R error': frob_norm_np(R - R_comp),
        'b1 error': vec_2_norm_np(b1_true - b1_comp),
        'b2 error': vec_2_norm_np(b2_true - b2_comp)
    }

    for name, error in errors.items():
        print(f'{name}: {error}')

    return x_comp, R_comp, b1_comp, b2_comp


if __name__ == "__main__":
    while True:
        x_comp, R_comp, b1_comp, b2_comp = least_squares_solver_interactive()

        user_input = input("\nRun another problem? (y/n) [default=n]: ").strip().lower()
        if user_input != 'y':
            break

    print("Thank you for using the Least Squares Solver!")