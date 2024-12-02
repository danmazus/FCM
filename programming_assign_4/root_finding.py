def f(x):
    f = pow(x, 3) + (-3*x) + 1
    return f

def p(x, rho, d):
    f = (x - rho)**d
    return f


# Regula-Falsi
def reg_fal_method(f, a0, b0, max_iter, tol = 1e-6):
    if f(a0) * f(b0) >= 0:
        raise ValueError("The function values at a0 and b0 must have opposite signs")

    # Setting x_0 as either a_0 or b_0
    if abs(f(a0)) < abs(f(b0)):
        x = a0
    else:
        x = b0
    #x = a0
    #x = b0

    # Setting initial values for iterations
    a_k = a0
    b_k = b0

    # Compute Initial q_0
    q_k = (f(b_k) - f(a_k)) / (b_k - a_k)

    # Iteration start
    k = 0

    # Finding root
    while k < max_iter:

        # Compute x_(k+1)
        x_next = x - (f(x) / q_k)

        # Check for convergence
        if abs(f(x_next)) < tol:
            return x_next, k + 1

        # Setting Next values depending on what the signs are
        if f(x_next) * f(a_k) < 0:
            a_k_next = a_k
            b_k_next = x_next
        elif f(x_next) * f(b_k) < 0:
            a_k_next = x_next
            b_k_next = b_k
        else:
            # This is a return statement as we have found the root directly
            return x_next, k + 1

        # Compute q_(k+1)
        q_next = (f(b_k_next) - f(a_k_next)) / (b_k_next - a_k_next)

        # Update Values for next iterations
        a_k = a_k_next
        b_k = b_k_next
        x = x_next
        q_k = q_next
        k += 1

    raise ValueError("Regula Falsi method did not converge within maximum number of iterations")



# Secant Method
def secant_method(f, x0, x1, max_iter, tol = 1e-6):

    k = 0
    x_k0 = x0    # x_(-1)
    x_k1 = x1    # x_0

    #if f(x_k0) * f(x_k1) >= 0:
     #   raise ValueError("The function values at x0 and x1 must have opposite signs")

    if (f(x_k0) * f(x_k1) < 0):
        while k < max_iter:
            # Compute q_k
            q_k = (f(x_k1) - f(x_k0)) / (x_k1 - x_k0)

            # Compute x_(k+1)
            x_k_next = x_k1 - f(x_k1) / q_k

            # Check for direct convergence
            if abs(f(x_k_next)) < tol:
                return x_k_next, k + 1

            # Check for x_(k+1) - x_k convergence
            if abs(x_k_next - x_k1) < tol:
                return x_k_next, k + 1

            # Update values for next iteration
            x_k0 = x_k1
            x_k1 = x_k_next
            k += 1

        raise ValueError("Secant method did not converge within the maximum number of iterations")



# Newton's Method
def newton_method(f, x0, max_iter, m = 1.0, tol = 1e-6, h = 1e-6):

    # Define the Numerical Approximation of the derivative
    def der(f, x, h):
        return (f(x+h) - f(x)) / h

    # Set initial values
    k = 0
    x = x0

    while k < max_iter:
        # Compute q_k = f'(x_k)
        q_k = der(f, x, h)

        # Makes sure derivative value is not too small for division
        if abs(q_k) < 1e-12:
            raise ValueError("Evaluated Derivative is too small at iteration {k}: x_k = {x}, q_k = {q_k}")

        # Comptue x_(k+1)
        x_next = x - (m * (f(x) / q_k))

        if abs(x_next) < tol or abs(x_next - x) < tol:
            return x_next, k + 1

        x = x_next
        k += 1

    raise ValueError("Newton's method did not converge within the maximum number of iterations. Last x_k = {x}, iteration = {k}")



# Steffenson's Method
def steff_method(f, x0, max_iter, tol = 1e-6):
    x = x0
    k = 0
    while k < max_iter:
        # First way of implementing steffenson's method
        theta = (f(x + f(x)) - f(x)) / f(x)
        g = x - (f(x) / theta)
        x_next = g

        # Second way of implementing steffenson's method
        # denominator = f(x + f(x)) - f(x)
        # if abs(denominator) < 1e-12:
        #     raise ValueError(f"Evaluated Derivative is too small at iteration {k}: x_k = {x}")
        # x_next = x - ((f(x)**2) / denominator)

        # Convergence check for both direct or x_k+1 - x_k convergence
        if abs(f(x_next)) < tol or abs(x_next - x) < tol:
            return x_next, k + 1

        x = x_next
        k += 1

    raise ValueError("Steffenson's method did not converge within the maximum number of iterations")



# Test Cases for f(x) = x^3 - 3x + 1 from in class example to test correctness
solution_reg, iteration_reg = reg_fal_method(f, 1, 2, max_iter=1000, tol=1e-6)
solution_sec, iteration_sec = secant_method(f, 1, 2, max_iter=1000, tol=1e-6)
solution_new, iteration_new = newton_method(f, x0 = 2, max_iter=1000, m=1.0, tol=1e-6, h=1e-6)
solution_steff, iteration_steff = steff_method(f, 2, max_iter=1000, tol=1e-6)

print(f"\nRoot for Regula Falsi is: {solution_reg}")
print(f"\nIterations for Regula Falsi is: {iteration_reg}")
print(f"\nRoot for Secant is: {solution_sec}")
print(f"\nIterations for Secant is: {iteration_sec}")
print(f"\nRoot for Newton is: {solution_new}")
print(f"\nIterations for Newton is: {iteration_new}")
print(f"\nRoot for Steffenson is: {solution_steff}")
print(f"\nIterations for Steffenson is: {iteration_steff}")

