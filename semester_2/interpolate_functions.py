import copy
import numpy as np
import matplotlib.pyplot as plt

def chebyshev_points(a, b, n, flag, dtype=np.float32):
    """
    Function to create Chebyshev Points of the First kind
    Inputs:
        n: number of points needed
        flag: Flag to indicate which Chebyshev Points should be created
            1 = Uniform Mesh
            2 = Chebyshev Points of the First Kind
            3 = Chebyshev Points of the Second Kind
    Outputs:
        x_mesh: mesh points with desired type
    """
    # Initializing the mesh
    x_mesh = np.zeros(n+1, dtype=dtype)


    if flag == 1:
        x_mesh = np.linspace(a, b, n+1, dtype=dtype)

    # Looping over to create the mesh points for Chebyshev Points of the First Kind
    elif flag == 2:
        for i in range(n+1):
            x_mesh[i] = dtype(0.5 * (b-a) * np.cos(((2*i + 1) * np.pi) / (2 * n + 2)) + 0.5*(b+a))

    # Looping over to create the mesh points for Chebyshev Points of the Second Kind
    elif flag == 3:
        for i in range(n+1):
            x_mesh[i] = dtype(0.5 * (b-a) * np.cos((i * np.pi)/ n) + 0.5 * (b+a))

    return x_mesh.astype(dtype)

def x_mesh_order(x_mesh, flag):
    """
    Function to order a given mesh in either increasing, decreasing, or Leja order
    Inputs:
        x_mesh: The vector of points to be sorted
        flag: Flag to indicate which ordering is to be used
            1 = Decreasing order of values in mesh
            2 = Increasing order of values in mesh
            3 = Leja order of values in mesh
    Outputs:
        x_mesh_order: Points in order
    """

    # Initializing terms and the ordered set of terms
    n = len(x_mesh) - 1
    x_mesh_order = copy.deepcopy(x_mesh)

    # Decreasing order of x-values in mesh
    if flag == 1:
        for i in range(n):
            for j in range(0, n-i-1):
                if x_mesh_order[j] < x_mesh_order[j+1]:
                    x_mesh_order[j], x_mesh_order[j+1] = x_mesh_order[j+1], x_mesh_order[j]

    # Increasing order of x-values in mesh
    elif flag == 2:
        for i in range(n):
            for j in range(0, n-i-1):
                if x_mesh_order[j] > x_mesh_order[j+1]:
                    x_mesh_order[j], x_mesh_order[j+1] = x_mesh_order[j+1], x_mesh_order[j]

    # Leja ordering of x-values in mesh
    elif flag == 3:
        # Picking x_0 as the max by getting the index of the max through argmax and then reordering terms
        x_max_index = np.argmax(np.abs(x_mesh_order))
        x_mesh_order[0], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[0]

        # Looping over Rest of the values in the mesh starting from index 1 to n-1
        for i in range(1, n):
            # Initializing the product to 1
            product = np.ones(n)

            # Looping over the remaining values needed for the product, i to n-1
            for j in range(i, n):
                for k in range(i): # Looping over the already selected points from before, i.e. 0 to i-1
                    product[j] *= abs(x_mesh_order[j] - x_mesh_order[k])


            # Indexing through to find the next point with max product and add i to get correct index
            # Adding i is needed as this will adjust the index in the subarray to the original array location
            x_max_index = np.argmax(np.abs(product[i:])) + i

            # Reordering mesh based on the point with max product
            x_mesh_order[i], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[i]


    return x_mesh_order

def coef_gamma(x_mesh, f, dtype=np.float32):
    """
    Function to calculate the gamma coefficients and have an evaluation of the function at the mesh points
    Inputs:
        x_mesh: The mesh points
        n: number of points needed
        f: Function to be evaluated/to evaluate the points
    Outputs:
        gamma_vec: The gamma coefficients stored in a vector
        func_val: The evaluated function value at the mesh points
    """
    n = len(x_mesh) - 1

    gamma_vec = np.ones(n+1, dtype=dtype)
    #func_val = np.zeros(n+1, dtype=dtype)

    # for i in range(n+1):
    #     func_val[i] = dtype(f(x_mesh[i]))
    func_val = f(x_mesh)


    for i in range(n+1):
        for j in range(n+1):
            if i != j:
                gamma_vec[i] *= dtype((x_mesh[i] - x_mesh[j]))

    gamma_vec = dtype(1/gamma_vec)
    return gamma_vec.astype(dtype), func_val.astype(dtype)

def coef_beta(x_mesh, f, flag, dtype=np.float32):
    """
    Function to calculate the beta coefficients for Barycentric 2 using either the recursive formula to
    get the coefficients, Chebyshev points of the First Kind, or Chebyshev Points of the Second Kind. This
    function will also evaluate the function at the mesh points
    Inputs:
        x_mesh: The mesh points
        n: number of points needed
        f: Function to be evaluated/to evaluate the points
        flag: Flag to indicate which beta coefficients should be created
            1 = Recursive style without Chebyshev
            2 = Using Chebyshev Points of the First Kind
            3 = Using Chebyshev Points of the Second Kind
    Outputs:
        beta_vec: The beta coefficients stored in a vector
        func_val: The evaluated function value at the mesh points
    """
    n = len(x_mesh) - 1

    beta_vec = np.zeros(n+1, dtype=dtype)

    func_val = f(x_mesh)

    # Using Uniform Mesh
    if flag == 1:
        beta_vec[0] = 1
        for i in range(n):
            beta_vec[i+1] = dtype(-beta_vec[i] * ((n - i) / (i + 1)))

    # Chebyshev Point of the First Kind
    elif flag == 2:
        for i in range(n+1):
            beta_vec[i] = dtype((-1)**i) * dtype(np.sin(((2*i + 1) * np.pi) / (2*n + 2)))

    # Chebyshev Points of the Second Kind
    else:
        for i in range(n+1):
            if i == 0 or i == n:
                beta_vec[i] = dtype(((-1) ** i) * (1 / 2))
            else:
                beta_vec[i] = dtype(((-1) ** i) * 1)

    return beta_vec, func_val.astype(dtype)

def bary_1_interpolation(gamma_vec, x_mesh, x_values, y, dtype=np.float32):
    """
    This function is implementing the Barycentric 1 form interpolation and evaluating the polynomial.
    Inputs:
        gamma_vec: Coefficient weights for p_(k-1)
        x_mesh: Given x-values
        x_values: x-values that are to be interpolated through/estimated
        y: The corresponding y-values associated with the mesh
        n: The length of the mesh minus 1

    Outputs:
        m_curr: The new Coefficient weights for p_k
        p_eval: the evaluated polynomial at the x_values
    """
    n = len(x_mesh) - 1

    ## EVALUATING POLYNOMIAL USING BARYCENTRIC 1
    p_eval = np.zeros(len(x_values), dtype=dtype)


    # condition_1 = np.zeros(len(x_values))
    # condition_y_numer = np.zeros(len(x_values))
    for j in range(len(x_values)):
        numerical_stab = np.isclose(x_values[j], x_mesh, atol=np.finfo(dtype).eps).any()

        if numerical_stab:
            p_eval[j] = y[np.argmin(np.abs(x_values[j] - x_mesh))]


        #omega = np.prod(x_values[j] - x_mesh, dtype=dtype)
        omega = dtype(1)
        for i in range(n+1):
            omega *= dtype((x_values[j] - x_mesh[i]))

        term = 0
        for i in range(n+1):
            term += dtype((y[i] * gamma_vec[i]) / (x_values[j] - x_mesh[i]))

        # sum_cond = 0
        # sum_cond_y = 0
        # for i in range(n + 1):
        #     sum_cond_l = (gamma_vec[i] * omega) / (x_values[j] - x_mesh[i])
        #     sum_cond += np.abs(sum_cond_l)
        #     sum_cond_ly = sum_cond_l * y[i]
        #     sum_cond_y += np.abs(sum_cond_ly)
        # condition_1[j] = sum_cond
        # condition_y_numer[j] = sum_cond_y

        p_eval[j] = dtype(omega * term)


    return p_eval.astype(dtype)

def bary_2_interpolation(beta_vec, x_mesh, x_values, y, dtype=np.float32):
    n = len(x_mesh) - 1

    p_eval = np.zeros(len(x_values), dtype=dtype)
    # condition_1 = np.zeros(len(x_values))
    # condition_y = np.zeros(len(x_values))

    for j in range(len(x_values)):
        numerical_stab = np.isclose(x_values[j], x_mesh, atol=np.finfo(dtype).eps).any()

        if numerical_stab:
            closest = np.argmin(np.abs(x_values[j] - x_mesh))
            p_eval[j] = y[closest]
            continue

        # numer = dtype(0)
        # denom = dtype(0)
        # sum_numer_cond_1 = 0
        # sum_denom_cond_1 = 0
        # sum_numer_cond_y = 0
        # sum_denom_cond_y = 0

        tau = 0
        sigma = 0

        for i in range(n+1):
            rho = beta_vec[i]/(x_values[j] - x_mesh[i])
            sigma += y[i] * rho
            tau += rho
            # numer += dtype((y[i] * beta_vec[i]) / (x_values[j] - x_mesh[i]))
            # denom += dtype(beta_vec[i] / (x_values[j] - x_mesh[i]))
            #
            # frac = beta_vec[i] / (x_values[j] - x_mesh[i])
            # frac_y = frac * y[i]
            # sum_numer_cond_1 += np.abs(frac)
            # sum_denom_cond_1 += frac
            # sum_denom_cond_1 = np.abs(sum_denom_cond_1)
            # sum_numer_cond_y += np.abs(frac_y)
            # sum_denom_cond_y += frac_y
            # sum_denom_cond_y = np.abs(sum_denom_cond_y)

        p_eval[j] = dtype(sigma / tau)
        #p_eval[j] = dtype(numer / denom)

        #condition_1[j] = sum_numer_cond_1 / sum_denom_cond_1
        #condition_y[j] = sum_numer_cond_y / sum_denom_cond_y

    # for j in range(len(x_values)):
    #
    #     sum_numer_cond_1 = 0
    #     sum_denom_cond_1 = 0
    #     sum_numer_cond_y = 0
    #     sum_denom_cond_y = 0
    #
    #     for i in range(n+1):
    #         frac = beta_vec[i] / (x_values[j] - x_mesh[i])
    #         frac_y = frac * y[i]
    #         sum_numer_cond_1 += np.abs(frac)
    #         sum_denom_cond_1 += frac
    #         sum_denom_cond_1 = np.abs(sum_denom_cond_1)
    #         sum_numer_cond_y += np.abs(frac_y)
    #         sum_denom_cond_y += frac_y
    #         sum_denom_cond_y = np.abs(sum_denom_cond_y)
    #
    #     condition_1[j] = sum_numer_cond_1 / sum_denom_cond_1
    #     condition_y[j] = sum_numer_cond_y / sum_denom_cond_y

    return p_eval

def newton_divdiff(x_mesh, f, dtype=np.float32):
    n = len(x_mesh)

    """Computing the mesh values using the given function"""
    func_val = f(x_mesh)


    div_coeff = copy.deepcopy(func_val)
    for i in range(1, n):
        for j in range(n-1, i-1, -1):
            div_coeff[j] = dtype((div_coeff[j] - div_coeff[j-1]) / (x_mesh[j] - x_mesh[j-i]))

    return func_val.astype(dtype), div_coeff.astype(dtype)

def horner_interpolation(x_mesh, x_values, div_coeff, dtype=np.float32):
    n = len(x_mesh) - 1
    alpha = copy.deepcopy(div_coeff)
    p_eval = np.zeros(len(x_values), dtype=dtype)
    for j in range(len(x_values)):
        s = alpha[-1]
        for i in range(n-1, -1, -1):
            s = dtype(s * (x_values[j] - x_mesh[i]) + alpha[i])

        #p_eval.append(s)
        p_eval[j] = s

    p_eval = np.array(p_eval, dtype=dtype)

    return p_eval

def product_func(x_values, x_mesh, alpha, dtype=np.float32):
    n = len(x_mesh) - 1
    d = []
    d[0] = alpha
    for j in range(len(x_values)):
        for i in range(1, n+1):
            d[i] = dtype(d[i-1] * (x_values[j] - x_mesh[i]))


    return np.array(d, dtype=dtype)

def hermpol(x, y, dy, z, dtype=np.float32):
    """
    HERMPOL Hermite Polynomial Interpolation
    Computes the Hermite interpolating polynomial of a function.
    Inputs:
    x: The interpolation nodes (x_mesh)
    y: The function values at x
    dy: The derivative values at x
    z: Contains the points at which the interpolating polynomial must be evaluated

    Outputs:
    herm: The Hermite interpolating polynomial
    """
    n = len(x)
    m = len(z)
    dtype_cast = dtype
    herm = []
    for j in range(m):
        xx = z[j]
        hxv = 0
        for i in range(n):
            den = dtype_cast(1)
            num = dtype_cast(1)
            xn = x[i]
            derLi = dtype_cast(0)
            for k in range(n):
                if k != i:
                    num = dtype_cast(num * (xx - x[k]))
                    arg = dtype_cast(xn - x[k])
                    den *= dtype_cast(arg)
                    derLi = dtype_cast(derLi + 1/arg)
            Lix2 = dtype_cast((num/den)**2)
            p = dtype_cast((1 - (2 * (xx-xn)*derLi)) * Lix2)
            q = dtype_cast((xx - xn) * Lix2)
            hxv = dtype_cast(hxv + (y[i] * p + dy[i] * q))

        herm.append(hxv)

    herm = np.array(herm, dtype=dtype)

    return herm


################## DR GALLIVAN VERSION OF CODES ######################

def bary1coeffs(xvals, n):
    """
    Computes the inverses of the coefficients for the barycentric form 1
    (modified Lagrange) of the interpolating polynomial on the mesh given
    in the (n+1)-length array xvals.

    Parameters:
        xvals (numpy array): Array of length (n+1) containing interpolation points.
        n (int): Degree of the interpolating polynomial.

    Returns:
        numpy array: Array of length (n+1) containing the computed coefficients.
    """

    np1 = n + 1
    gammainvs = np.zeros(np1)

    # Initialize to the linear coefficients
    gammainvs[0] = xvals[0] - xvals[1]  # (x_0 - x_1)
    gammainvs[1] = xvals[1] - xvals[0]  # (x_1 - x_0)

    # Degree for whom the parameters are in gammainvs
    d = 1

    # Compute coefficients from degree 2 to n
    for d in range(2, n + 1):
        p = 1.0
        for i in range(d):
            t = xvals[i] - xvals[d]
            gammainvs[i] *= t  # m_i^d = t * m_i^{d-1}
            p = -p * t
        gammainvs[d] = p

    return gammainvs


def bary2coeffs(n, pointtype):
    """
    Computes the coefficients for the barycentric form 2.

    Parameters:
        n (int): Degree of the interpolating polynomial.
        pointtype (int): Type of point set.
            0 - Uniform mesh
            1 - Chebyshev 1st kind
            2 - Chebyshev 2nd kind

    Returns:
        numpy array: Array of length (n+1) containing the computed coefficients.
    """

    np1 = n + 1
    beta = np.zeros(np1)

    if pointtype == 0:
        # Uniform mesh Barycentric 2 coefficients
        beta[0] = 1.0
        for j in range(1, np1):
            beta[j] = -beta[j - 1] * ((n - j + 1) / j)

    elif pointtype == 1:
        # Chebyshev of the 1st kind Barycentric 2 coefficients
        for j in range(np1):
            beta[j] = (-1.0) ** j * np.sin(((2.0 * j + 1.0) * np.pi) / (2.0 * n + 2))

    elif pointtype == 2:
        # Chebyshev of the 2nd kind Barycentric 2 coefficients
        beta = np.array([0.5] + [1.0] * (n - 1) + [0.5]) * (-1) ** np.arange(np1)

    else:
        print(f"Invalid pointset type = {pointtype}")
        return None  # Return None to indicate an error condition

    return beta


def nddcoeffs(xvals_arg, yvals_arg, n):
    """
    Computes the Newton divided difference coefficients.

    Parameters:
        xvals_arg (numpy array): Array of x-values.
        yvals_arg (numpy array): Array of corresponding y-values.
        n (int): Degree of the interpolating polynomial.

    Returns:
        numpy array: Array of length (n+1) containing the Newton divided difference coefficients.
    """

    np1 = n + 1
    ds = np.zeros_like(xvals_arg, dtype=float)
    divdiffs = np.zeros_like(xvals_arg, dtype=float)

    # ---------------------------------------------
    # Initialize to the linear coefficients
    # ---------------------------------------------
    ds[0] = yvals_arg[0] / (xvals_arg[0] - xvals_arg[1])  # y_0 / (x_0 - x_1)
    ds[1] = yvals_arg[1] / (xvals_arg[1] - xvals_arg[0])  # y_1 / (x_1 - x_0)
    divdiffs[0] = yvals_arg[0]  # y_0
    divdiffs[1] = ds[0] + ds[1]  # y[x_0, x_1]

    # ---------------------------------------------
    # From degree 2 to n
    # Add effect of x_d, i.e., xvals_arg[d]
    # to ds
    # ---------------------------------------------
    for d in range(2, n + 1):
        p = 1.0
        s = 0.0
        for i in range(d):
            t = xvals_arg[i] - xvals_arg[d]  # (x_i - x_d)
            ds[i] /= t  # d_i / (x_i - x_d)
            p *= t
            s += ds[i]

        ds[d] = ((-1.0) ** d) * (yvals_arg[d] / p)
        divdiffs[d] = s + ds[d]  # f[x_0, ..., x_d]

    return divdiffs


def bary1eval(xvals, yvals, gammainvs, n, xcheck, ncheck, kappa_numer, lebesgue_vals):
    """
    Evaluates the interpolating polynomial p(x) of degree n in barycentric form 1.

    Parameters:
        xvals (numpy array): Interpolation nodes.
        yvals (numpy array): Function values at the nodes.
        gammainvs (numpy array): Barycentric weights (inverse gamma).
        n (int): Degree of the interpolating polynomial.
        xcheck (numpy array): Evaluation points.
        ncheck (int): Number of evaluation points.
        kappa_numer (numpy array): Numerator of kappa(x, n, f) (to be updated).
        lebesgue_vals (numpy array): Lebesgue function values (to be updated).

    Returns:
        tuple: (pvals, kappa_numer, lebesgue_vals)
    """

    omega = np.ones(ncheck)
    exact = np.zeros(ncheck, dtype=int)
    onevals = np.ones(ncheck)
    np1 = n + 1

    for i in range(np1):
        xdiff = xcheck - onevals * xvals[i]
        exact[xdiff == 0] = i + 1  # Store index (MATLAB is 1-based, so adjust for Python)
        omega *= xdiff

    # Initialize output arrays
    pvals = np.zeros_like(xcheck)
    kappa_numer = np.zeros_like(xcheck)
    lebesgue_vals = np.zeros_like(xcheck)

    for i in range(np1):
        xdiff = xcheck - onevals * xvals[i]
        term = (onevals * (yvals[i] / gammainvs[i])) * (omega / xdiff)
        pvals += term
        kappa_numer += np.abs(term)
        lebesgue_vals += np.abs((onevals / gammainvs[i]) * (omega / xdiff))

    # Fix values where xcheck matches xvals (avoiding NaNs)
    jfixes = np.where(exact > 0)[0]  # Get indices where exact matches
    pvals[jfixes] = yvals[exact[jfixes] - 1]  # Adjust for zero-based indexing
    kappa_numer[jfixes] = np.abs(yvals[exact[jfixes] - 1])
    lebesgue_vals[jfixes] = np.abs(yvals[exact[jfixes] - 1])

    return pvals, kappa_numer, lebesgue_vals


def bary2eval(xvals, yvals, beta, n, xcheck, ncheck):
    """
    Evaluates the interpolating polynomial p(x) of degree n in barycentric form 2.

    Parameters:
        xvals (numpy array): Interpolation nodes.
        yvals (numpy array): Function values at the nodes.
        beta (numpy array): Barycentric weights.
        n (int): Degree of the interpolating polynomial.
        xcheck (numpy array): Evaluation points.
        ncheck (int): Number of evaluation points.

    Returns:
        numpy array: Evaluated polynomial values at xcheck.
    """

    exact = np.zeros(ncheck, dtype=int)
    np1 = n + 1

    for i in range(np1):
        xdiff = xcheck - xvals[i]
        exact[xdiff == 0] = i + 1  # Store index (MATLAB is 1-based, so adjust for Python)

    # Initialize numerator and denominator
    numer = np.zeros_like(xcheck)
    denom = np.zeros_like(xcheck)
    pvals = np.zeros_like(xcheck)

    for i in range(np1):
        xdiff = xcheck - xvals[i]
        temp = beta[i] / xdiff
        numer += temp * yvals[i]
        denom += temp

    pvals = numer / denom

    # Fix values where xcheck matches xvals (avoiding division by zero)
    jfixes = np.where(exact > 0)[0]  # Get indices where exact matches
    pvals[jfixes] = yvals[exact[jfixes] - 1]  # Adjust for zero-based indexing

    return pvals


def nddeval(xvals, yvals, divdiffs, n, xcheck, ncheck):
    """
    Evaluates the interpolating polynomial using the modified Horner's rule.

    Parameters:
        xvals (numpy array): Interpolation nodes.
        yvals (numpy array): Function values at the nodes.
        divdiffs (numpy array): Divided differences coefficients.
        n (int): Degree of the interpolating polynomial.
        xcheck (numpy array): Evaluation points.
        ncheck (int): Number of evaluation points.

    Returns:
        numpy array: Evaluated polynomial values at xcheck.
    """

    np1 = n + 1
    exact = np.zeros(ncheck, dtype=int)
    pvals = np.zeros(ncheck)

    # Identify exact matches
    for i in range(np1):
        xdiff = xcheck - xvals[i]
        exact[xdiff == 0] = i + 1  # Convert MATLAB 1-based indexing to Python 0-based

    # Evaluate polynomial using modified Horner's rule
    for k in range(ncheck):
        t = divdiffs[np1 - 1]  # MATLAB index np1 -> Python index np1-1
        for i in range(n - 1, -1, -1):
            t = (xcheck[k] - xvals[i]) * t + divdiffs[i]
        pvals[k] = t

    # Fix values at exact matches
    jfixes = np.where(exact > 0)[0]  # Get indices where exact matches
    pvals[jfixes] = yvals[exact[jfixes] - 1]  # Adjust for zero-based indexing

    return pvals


def pointsetordering(xvals, permout, np1, iperm):
    """
    Reorder interpolation pointset

    Parameters:
    - xvals: Input x values
    - permout: Permutation indices
    - np1: Number of points
    - iperm: Permutation type

    Returns:
    - Reordered x values
    - Updated permutation indices
    """
    if iperm == 0:  # Fast Leja order (not fully implemented)
        pass
    elif iperm == 1:  # Increasing order
        indices = np.argsort(xvals)
        xvals = xvals[indices]
        permout = permout[indices]
    elif iperm == -1:  # Decreasing order
        indices = np.argsort(xvals)[::-1]
        xvals = xvals[indices]
        permout = permout[indices]

    return xvals, permout


#########################################################################


class PolynomialInterpolation:
    def __init__(self, a, b, n, M, d, dtype):
        """
        Defining global variables that are used throughout the class

        a, b: Global interval of [a,b]
        n: Number of mesh points for non-piecewise interpolation
        M: Number of subintervals for piecewise interpolation
        d: Degree of piecewise interpolation, 1 (Linear), 2 (Quadratic), or 3 (Cubic)
        dtype: Precision to be used
        """

        self.a = a  # Left end point of global interval
        self.b = b  # Right end point of global interval
        self.n = n  # Global number of mesh points to be used if not piecewise interpolation
        self.M = M  # Number of subintervals to be used for piecewise interpolation
        self.d = d  # Degree using piecewise interpolation
        self.dtype = dtype  # Specifies what precision to be used

        self.x_mesh = None  # Setting x_mesh to be used
        self.div_coeff = None   # Setting the coefficients for Newton
        self.subintervals = None

    def local_mesh(self, flag):
        """
        Generates mesh points on the reference interval of [-1,1] with d + 1 mesh points defining
        the amount needed. Flag indicates which mesh type to use, 1 (Uniform), 2 (Chebyshev Points of the First Kind),
        or 3 (Chebyshev Points of the Second Kind). There is a ValueError to control that 1, 2, or 3 must be selected.
        This is used for piecewise interpolation only.
        """

        # Uniform
        if flag == 1:
            return np.linspace(-1, 1, self.d + 1)

        # Chebyshev Points of the First Kind
        elif flag == 2:
            return np.cos(((2 * np.arange(self.d + 1) + 1) * np.pi) / (2 * (self.d + 1) + 2))

        # Chebyshev Points of the Second Kind
        elif flag == 3:
            return np.cos((np.arange(self.d + 1) * np.pi) / (self.d))

        # If anything else is selected, error is raised
        else:
            raise ValueError("Invalid flag. Choose 1 (Uniform), 2 (Chebyshev Points of the First Kind), or 3 (Chebyshev Points of the Second Kind).")

    def mesh_points(self, flag, piecewise=False):
        """
        Generates mesh points on either the interval [a,b] and scales it accordingly for Chebyshev Points or considers
        if piecewise interpolation is used by either True or False. If false, this generates a mesh over [a,b]. If True,
        generates mesh using the local_mesh function above. As the boolean is true, this means that piecewise interpolation
        is being used.
        """

        # Defining the mesh for nonpiecewise interpolation
        if not piecewise:
            if flag == 1:
                self.x_mesh = np.linspace(self.a, self.b, self.n + 1, dtype=self.dtype)

            # Looping over to create the mesh points for Chebyshev Points of the First Kind
            elif flag == 2:
                self.x_mesh = np.zeros(self.n + 1)
                for i in range(self.n + 1):
                    self.x_mesh[i] = self.dtype(0.5 * (self.b - self.a) * np.cos(((2 * i + 1) * np.pi) / (2 * self.n + 2)) + 0.5 * (self.b + self.a))

            # Looping over to create the mesh points for Chebyshev Points of the Second Kind
            elif flag == 3:
                self.x_mesh = np.zeros(self.n + 1)
                for i in range(self.n + 1):
                    self.x_mesh[i] = self.dtype(0.5 * (self.b - self.a) * np.cos((i * np.pi) / self.n) + 0.5 * (self.b + self.a))

        # Defining mesh for piecewise interpolation
        else:
            # Create the M subintervals from a to b
            global_mesh = np.linspace(self.a, self.b, self.M + 1, dtype=self.dtype)

            # Defining the amount of mesh points needed, Mk + 1
            self.x_mesh = np.zeros(self.M * (self.d + 1) + 1, dtype=self.dtype)

            for s in range(self.M):
                # Getting the endpoints of the given subinterval from global_mesh
                x_s = global_mesh[s]
                x_s_1 = global_mesh[s + 1]

                # Compute the width of the given interval, H_s = x_s_1 - x_s
                H_s = x_s_1 - x_s

                # Grab the local mesh points from the function generating locally
                local_x_mesh = self.local_mesh(flag)

                # Map the local mesh points to the given subinterval
                x_mid = (x_s + x_s_1) / 2   # Getting the midpoint of the interval
                global_x_mesh = x_mid + local_x_mesh * (H_s / 2)    # Mapping of the local mesh points to subinterval

                # Slicing of x_mesh so that each subinterval is added from "left to right" from the computed global x_mesh
                self.x_mesh[s * (self.d + 1): (s + 1) * (self.d + 1)] = global_x_mesh

            # Ensuring the last endpoint is b
            self.x_mesh[-1] = self.b

        return self.x_mesh.astype(self.dtype)

    def ordered_mesh(self, flag):
        # Initializing terms and the ordered set of terms
        x_mesh_order = copy.deepcopy(self.x_mesh)
        n = len(x_mesh_order) - 1

        # Decreasing order of x-values in mesh
        if flag == 1:
            for i in range(n):
                for j in range(0, n - i - 1):
                    if x_mesh_order[j] < x_mesh_order[j + 1]:
                        x_mesh_order[j], x_mesh_order[j + 1] = x_mesh_order[j + 1], x_mesh_order[j]

        # Increasing order of x-values in mesh
        elif flag == 2:
            for i in range(n):
                for j in range(0, n - i - 1):
                    if x_mesh_order[j] > x_mesh_order[j + 1]:
                        x_mesh_order[j], x_mesh_order[j + 1] = x_mesh_order[j + 1], x_mesh_order[j]

        # Leja ordering of x-values in mesh
        elif flag == 3:
            # Picking x_0 as the max by getting the index of the max through argmax and then reordering terms
            x_max_index = np.argmax(np.abs(x_mesh_order))
            x_mesh_order[0], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[0]

            # Looping over Rest of the values in the mesh starting from index 1 to n-1
            for i in range(1, n):
                # Initializing the product to 1
                product = np.ones(n)

                # Looping over the remaining values needed for the product, i to n-1
                for j in range(i, n):
                    for k in range(i):  # Looping over the already selected points from before, i.e. 0 to i-1
                        product[j] *= abs(x_mesh_order[j] - x_mesh_order[k])

                # Indexing through to find the next point with max product and add i to get correct index
                # Adding i is needed as this will adjust the index in the subarray to the original array location
                x_max_index = np.argmax(np.abs(product[i:])) + i

                # Reordering mesh based on the point with max product
                x_mesh_order[i], x_mesh_order[x_max_index] = x_mesh_order[x_max_index], x_mesh_order[i]

        return x_mesh_order

    def gamma_coefficients(self, f):
        n = len(self.x_mesh) - 1

        gamma_vec = np.ones(n + 1, dtype=self.dtype)

        func_val = f(self.x_mesh)

        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    gamma_vec[i] *= self.dtype((self.x_mesh[i] - self.x_mesh[j]))

        gamma_vec = self.dtype(1 / gamma_vec)

        return gamma_vec.astype(self.dtype), func_val.astype(self.dtype)

    def beta_coefficients(self, f, flag):
        n = len(self.x_mesh) - 1

        beta_vec = np.zeros(n + 1, dtype=self.dtype)

        func_val = f(self.x_mesh)

        # Using Uniform Mesh
        if flag == 1:
            beta_vec[0] = 1
            for i in range(n):
                beta_vec[i + 1] = self.dtype(-beta_vec[i] * ((n - i) / (i + 1)))

        # Chebyshev Point of the First Kind
        elif flag == 2:
            for i in range(n + 1):
                beta_vec[i] = self.dtype((-1) ** i) * self.dtype(np.sin(((2 * i + 1) * np.pi) / (2 * n + 2)))

        # Chebyshev Points of the Second Kind
        else:
            for i in range(n + 1):
                if i == 0 or i == n:
                    beta_vec[i] = self.dtype(((-1) ** i) * (1 / 2))
                else:
                    beta_vec[i] = self.dtype(((-1) ** i) * 1)

        return beta_vec, func_val.astype(self.dtype)

    def bary_1_interpolation(self, gamma_vec, x_values, func_vals):
        n = len(self.x_mesh) - 1

        ## EVALUATING POLYNOMIAL USING BARYCENTRIC 1
        p_eval = np.zeros(len(x_values), dtype=self.dtype)

        for j in range(len(x_values)):
            numerical_stab = np.isclose(x_values[j], self.x_mesh, atol=np.finfo(self.dtype).eps).any()

            if numerical_stab:
                p_eval[j] = func_vals[np.argmin(np.abs(x_values[j] - self.x_mesh))]

            omega = self.dtype(1)
            for i in range(n + 1):
                omega *= self.dtype((x_values[j] - self.x_mesh[i]))

            term = 0
            for i in range(n + 1):
                term += self.dtype((func_vals[i] * gamma_vec[i]) / (x_values[j] - self.x_mesh[i]))

            p_eval[j] = self.dtype(omega * term)

        return p_eval.astype(self.dtype)

    def bary_2_interpolation(self, beta_vec, x_values, func_vals):
        n = len(self.x_mesh) - 1

        p_eval = np.zeros(len(x_values), dtype=self.dtype)

        for j in range(len(x_values)):
            numerical_stab = np.isclose(x_values[j], self.x_mesh, atol=np.finfo(self.dtype).eps).any()

            if numerical_stab:
                closest = np.argmin(np.abs(x_values[j] - self.x_mesh))
                p_eval[j] = func_vals[closest]
                continue

            tau = 0
            sigma = 0

            for i in range(n + 1):
                rho = beta_vec[i] / (x_values[j] - self.x_mesh[i])
                sigma += func_vals[i] * rho
                tau += rho


            p_eval[j] = self.dtype(sigma / tau)

        return p_eval

    # def newton_divdiff(self, f, piecewise=False):
    #     if not piecewise:
    #         m = len(self.x_mesh)
    #
    #         """Computing the mesh values using the given function"""
    #         func_val = f(self.x_mesh)
    #
    #         div_coeff = copy.deepcopy(func_val)
    #         for i in range(1, m):
    #             for j in range(m - 1, i - 1, -1):
    #                 div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (self.x_mesh[j] - self.x_mesh[j - i]))
    #
    #         self.div_coeff = div_coeff.astype(self.dtype)
    #
    #         return self.div_coeff, func_val.astype(self.dtype)
    #
    #     else:
    #         self.div_coeff = []
    #         self.subintervals = []
    #
    #         for s in range(self.M):
    #             start = s * (self.d + 1)
    #             end = (s + 1) * (self.d + 1)
    #
    #             x_sub = self.x_mesh[start:end]
    #             f_sub = f(x_sub)
    #
    #             div_coeff = copy.deepcopy(f_sub)
    #             for i in range(1, len(x_sub)):
    #                 for j in range(len(x_sub) - 1, i - 1, -1):
    #                     div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (x_sub[j] - x_sub[j - i]))
    #
    #             self.div_coeff.append(div_coeff)
    #             self.subintervals.append(x_sub)
    #
    #             # print(f"Subinterval {s}: x_sub = {x_sub}")  # Debugging output
    #             # print(f"Divided Differences {s}: {div_coeff}\n")  # Debugging output
    #
    #         return self.div_coeff

    def newton_divdiff(self, f, df, piecewise=False, hermite=False):
        if not hermite:
            if not piecewise:
                ## Newton Divided Difference ##
                m = len(self.x_mesh)

                """Computing the mesh values using the given function"""
                func_val = f(self.x_mesh)

                div_coeff = copy.deepcopy(func_val)
                for i in range(1, m):
                    for j in range(m - 1, i - 1, -1):
                        div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (self.x_mesh[j] - self.x_mesh[j - i]))

                self.div_coeff = div_coeff.astype(self.dtype)

                return self.div_coeff, func_val.astype(self.dtype)

            else:
                ## Piecewise newton divided difference ##
                self.div_coeff = []
                self.subintervals = []

                for s in range(self.M):
                    # Setting left endpoint
                    start = s * (self.d + 1)

                    # Setting right endpoint
                    end = (s + 1) * (self.d + 1)

                    # Getting the mesh points in the given subinterval
                    x_sub = self.x_mesh[start:end]

                    # Calculating the y-values of given subinterval
                    f_sub = f(x_sub)

                    div_coeff = copy.deepcopy(f_sub)
                    for i in range(1, len(x_sub)):
                        for j in range(len(x_sub) - 1, i - 1, -1):
                            div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (x_sub[j] - x_sub[j - i]))

                    self.div_coeff.append(div_coeff)
                    self.subintervals.append(x_sub)

                    # print(f"Subinterval {s}: x_sub = {x_sub}")  # Debugging output
                    # print(f"Divided Differences {s}: {div_coeff}\n")  # Debugging output

                return self.div_coeff

        else:
            ## Piecewise Hermite Divided Difference Coefficients ##
            self.div_coeff = []
            self.subintervals = []

            for s in range(self.M):
                # Setting left endpoint
                start = s * (self.d + 1)

                # Setting right endpoint
                end = (s + 1) * (self.d + 1)

                # Getting the mesh points in the given subinterval
                x_sub = self.x_mesh[start:end]

                # Calculating y-values in given subinterval
                y_sub = f(x_sub)

                # Getting the derivative values associated with the given subinterval
                deriv_sub = df(x_sub)


                # Calculating the hermite divided difference table for the subinterval
                herm_table = self.hermite_divdiff(y_sub, deriv_sub, x_sub)

                self.div_coeff.append(herm_table)

                self.subintervals.append(x_sub)

            return self.div_coeff

    def get_divided_diff(self):
        return self.div_coeff

    def horner_interpolation(self, x_values):
        n = len(self.x_mesh) - 1
        alpha = copy.deepcopy(self.div_coeff)
        p_eval = np.zeros(len(x_values), dtype=self.dtype)

        for j in range(len(x_values)):
            s = alpha[-1]
            for i in range(n - 1, -1, -1):
                s = self.dtype(s * (x_values[j] - self.x_mesh[i]) + alpha[i])

            # p_eval.append(s)
            p_eval[j] = s

        p_eval = np.array(p_eval, dtype=self.dtype)

        return p_eval

    def piecewise_interpolation(self, x, f, df, flag, hermite=False):
        """
        Evaluates the piecewise polynomial interpolation at a given x using Newton's divided differences.
        """

        global s

        if flag == 3:
            sorted_x_mesh = sorted(self.x_mesh)

        else:
            sorted_x_mesh = self.x_mesh

        # Find the absolute bounds of the mesh
        mesh_min = sorted_x_mesh[0]
        mesh_max = sorted_x_mesh[-1]

        # Handle case where x is outside the entire mesh domain
        if x < mesh_min:
            # Option 1: Return None
            return None

            # Option 2: Use the first subinterval for extrapolation
            # s = 0
        elif x > mesh_max:
            # Option 1: Return None
            return None

            # Option 2: Use the last subinterval for extrapolation
            #s = self.M - 1
        else:
            # Find which subinterval contains x
            for s in range(self.M):
                x_s = sorted_x_mesh[s * (self.d + 1)]
                x_s_1 = sorted_x_mesh[(s + 1) * (self.d + 1)] if s < self.M - 1 else sorted_x_mesh[-1]

                if x_s <= x <= x_s_1:
                    break
            else:
                return None


        # Compute the Newton polynomial for this subinterval
        if not hermite:
            idx_start = s * (self.d + 1)
            #x_submesh = sorted_x_mesh[idx_start: idx_start + self.d + 1]
            x_submesh = self.x_mesh[idx_start: idx_start + self.d + 1]
            coeffs = self.div_coeff[s]

            # Perform Newton's nested evaluation
            result = coeffs[-1]
            for i in range(self.d - 1, -1, -1):
                result = result * (x - x_submesh[i]) + coeffs[i]

        else:
            idx_start = s * (self.d + 1)
            x_submesh = sorted_x_mesh[idx_start: idx_start + self.d + 1]

            a_i, b_i = x_submesh[0], x_submesh[-1]
            f_a, f_b = f(a_i), f(b_i)
            df_a, df_b = df(a_i), df(b_i)

            h = b_i - a_i

            third_divdiff = (f_b - f_a) / h**2 - df_a / h
            fourth_divdiff = (df_b + df_a) / h**2 - 2*(f_b - f_a)/ h**3

            result = f_a + df_a * (x - a_i) + third_divdiff * (x - a_i)**2 + fourth_divdiff * (x - a_i)**2*(x - b_i)


        return self.dtype(result)

    # def cubic_spline(self, f, boundary = 2):
    #     n = len(self.x_mesh) - 1
    #
    #     y = f(self.x_mesh)
    #
    #     # Step Sizes h_i
    #     h = np.zeros(n, dtype=self.dtype)
    #     for i in range(1, n):
    #         h[i] = self.dtype(self.x_mesh[i+1] - self.x_mesh[i])
    #
    #     # Setup of the tridiagonal matrix for second derivatives
    #     # Matrix is composed of elements mu, b, lamb
    #     mu = np.zeros(n+1, dtype=self.dtype)
    #     b = np.zeros(n+1, dtype=self.dtype)
    #     lamb = np.zeros(n+1, dtype=self.dtype)
    #     d = np.zeros(n+1, dtype=self.dtype)
    #
    #     # Interior points
    #     for i in range(1, n):
    #         mu[i] = self.dtype(h[i-1] / (h[i-1] + h[i]))
    #         b[i] = self.dtype(2.0)
    #         lamb[i] = self.dtype(h[i] / (h[i-1] + h[i]))
    #
    #     # Boundary Conditions
    #     if boundary == 2:
    #         # Second derivative matches
    #         b[0] = self.dtype(1.0)
    #         lamb[0] = self.dtype(0.0)
    #         d[0] = self.dtype(0.0)
    #
    #         mu[n] = self.dtype(0.0)
    #         b[n] = self.dtype(1.0)
    #         d[n] = self.dtype(0.0)
    #
    #     second_derivatives = self.solve_TriDiag(mu, b, lamb, d)
    #
    #     coefs = []
    #     for i in range(n):
    #         mu_i = y[i]
    #         b_i = (y[i+1] - y[i]) / h[i] - h[i] * (2 * second_derivatives[i] + second_derivatives[i + 1]) / 6
    #         lamb_i = second_derivatives[i] / 2
    #         d_i = (second_derivatives[i+1] - second_derivatives[i]) / (6 * h[i])
    #
    #         coefs.append((self.dtype(mu_i), self.dtype(b_i), self.dtype(lamb_i), self.dtype(d_i)))
    #
    #     return coefs
    #
    # def solve_TriDiag(self, mu, b, lamb, d):
    #     n = len(b) - 1
    #     x = np.zeros(n+1, dtype=self.dtype)
    #
    #     for i in range(1, n+1):
    #         m = self.dtype(mu[i] / b[i-1])
    #         b[i] = self.dtype(b[i] - m * lamb[i-1])
    #         d[i] = self.dtype(d[i] - m * d[i-1])
    #
    #     x[n] = self.dtype(d[n] / b[n])
    #     for i in range(n - 1, -1, -1):
    #         x[i] = self.dtype((d[i] - lamb[i] * x[i+1]) / b[i])
    #
    #
    #     return x
    #
    # def evaluate_spline(self, x_eval, coefs):
    #     if x_eval < self.x_mesh[0] or x_eval > self.x_mesh[-1]:
    #         raise ValueError(f'Evaluation point {x_eval} outside data range [{self.x_mesh[0]}, {self.x_mesh[-1]}]')
    #
    #     n = len(self.x_mesh) - 1
    #     i = 0
    #     while i < n and x_eval > self.x_mesh[i+1]:
    #         i += 1
    #
    #     mu, b, lamb, d = coefs[i]
    #
    #     dx = self.dtype(x_eval - self.x_mesh[i])
    #
    #     return self.dtype(mu + b * dx + lamb * dx**2 + d * dx**3)
    #
    # def evaluate_spline_deriv(self, x_eval, coefs, order=1):
    #     if order not in [1, 2]:
    #         raise ValueError(f'Order {order} not supported.')
    #
    #     if x_eval < self.x_mesh[0] or x_eval > self.x_mesh[-1]:
    #         raise ValueError(f'Evaluation point {x_eval} outside data range [{self.x_mesh[0]}, {self.x_mesh[-1]}]')
    #
    #     n = len(self.x_mesh) - 1
    #     i = 0
    #     while i < n and x_eval > self.x_mesh[i+1]:
    #         i += 1
    #
    #     mu, b, lamb, d = coefs[i]
    #
    #     dx = self.dtype(x_eval - self.x_mesh[i])
    #
    #     if order == 1:
    #         return self.dtype(b + 2 * lamb * dx + 3 * d * dx**2)
    #     else:
    #         return self.dtype(2 * lamb + 6 * d * dx)

