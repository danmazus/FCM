import numpy as np

from my_package import vec_2_norm


def compute_house(v, nj):
    """
    Compute Householder vector u and scalar rho.
    This function transforms a vector 'v', input, into a form of Hv = rho * e_1
    using Parlett's approach.

    Parameters:
        v (numpy array): Vector to be transformed
        nj (int): Number of elements of 'v' to be considered

    Returns:
        u (numpy array): Householder vector
        rho (float): Scalar value for transformed vector
    """
    # Step 1: Copy first 'nj' elements of 'v' into 'w'
    w = np.copy(v[:nj])

    # Step 2: Compute mu, which is the sum of squares of elements from the second element to the nj-th element in 'v'
    mu = np.sum(v[1:nj] ** 2)

    # Step 3: Compute rho, the 2-norm of 'v'
    rho = np.sqrt(v[0]**2 + mu)

    # Step 4: Update the first element of 'w' based on the sign of 'v[0]'
    if v[0] <= 0:
        w[0] -= rho
    else:
        w[0] = -mu / (v[0] - rho)

    # Step 5: Compute the 2-norm of w
    w2norm = vec_2_norm(w)

    # Step 6: Normalize 'w' to get the Householder vector 'u'
    u = w / w2norm

    return u, rho