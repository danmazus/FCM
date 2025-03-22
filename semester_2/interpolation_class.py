import numpy as np
import copy

class PolynomialInterpolation:
    def __init__(self, a, b, d, dtype, M=None, x_mesh=None, y_values=None):
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
        #self.n = n  # Global number of mesh points to be used if not piecewise interpolation
        self.M = M if M is not None else None  # Number of subintervals to be used for piecewise interpolation
        self.d = d  # Degree using piecewise interpolation
        self.dtype = dtype  # Specifies what precision to be used

        self.x_mesh = x_mesh if x_mesh is not None else None # Setting x_mesh to be used
        self.y_values = y_values if y_values is not None else None # Setting y_values incase given
        self.div_coeff = None  # Setting the coefficients for Newton
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
            raise ValueError(
                "Invalid flag. Choose 1 (Uniform), 2 (Chebyshev Points of the First Kind), or 3 (Chebyshev Points of the Second Kind).")

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
                self.x_mesh = np.linspace(self.a, self.b, self.d + 1, dtype=self.dtype)

            # Looping over to create the mesh points for Chebyshev Points of the First Kind
            elif flag == 2:
                self.x_mesh = np.zeros(self.d + 1)
                for i in range(self.d + 1):
                    self.x_mesh[i] = self.dtype(
                        0.5 * (self.b - self.a) * np.cos(((2 * i + 1) * np.pi) / (2 * self.d + 2)) + 0.5 * (
                                    self.b + self.a))

            # Looping over to create the mesh points for Chebyshev Points of the Second Kind
            elif flag == 3:
                self.x_mesh = np.zeros(self.d + 1)
                for i in range(self.d + 1):
                    self.x_mesh[i] = self.dtype(
                        0.5 * (self.b - self.a) * np.cos((i * np.pi) / self.d) + 0.5 * (self.b + self.a))

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
                x_mid = (x_s + x_s_1) / 2  # Getting the midpoint of the interval
                global_x_mesh = x_mid + local_x_mesh * (H_s / 2)  # Mapping of the local mesh points to subinterval

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
            numerical_stab = np.isclose(x_values[j], self.x_mesh, atol=np.finfo(self.dtype).eps)

            if numerical_stab.any():
                p_eval[j] = func_vals[np.argmin(np.abs(x_values[j] - self.x_mesh))]
                continue

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

    def newton_divdiff(self, f, piecewise=False, specify=False):
        if not piecewise:
            ## Newton Divided Difference ##
            m = len(self.x_mesh)

            """Computing the mesh values using the given function"""
            func_val = f(self.x_mesh)

            div_coeff = copy.deepcopy(func_val)
            for i in range(1, m):
                for j in range(m - 1, i - 1, -1):
                    div_coeff[j] = self.dtype(
                        (div_coeff[j] - div_coeff[j - 1]) / (self.x_mesh[j] - self.x_mesh[j - i]))

            self.div_coeff = div_coeff.astype(self.dtype)

            return self.div_coeff, func_val.astype(self.dtype)

        else:
            ## Piecewise newton divided difference ##
            self.div_coeff = []
            self.subintervals = []

            if not specify:
                for s in range(self.M + 1):
                    # Setting left endpoint
                    start = s * (self.d + 1)

                    # Setting right endpoint
                    end = (s + 1) * (self.d + 1)

                    # Getting the mesh points in the given subinterval
                    x_sub = self.x_mesh[start:end]

                    # Calculating the y-values of given subinterval
                    if self.y_values is None:
                        f_sub = f(x_sub)
                    else:
                        f_sub = self.y_values[start:end]

                    div_coeff = copy.deepcopy(f_sub)
                    for i in range(1, len(x_sub)):
                        for j in range(len(x_sub) - 1, i - 1, -1):
                            div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (x_sub[j] - x_sub[j - i]))

                    self.div_coeff.append(div_coeff)
                    self.subintervals.append(x_sub)

            else:
                for s in range(self.M + 1):
                    start = s
                    end = s + (self.d + 1)

                    x_sub = self.x_mesh[start:end]

                    if len(x_sub) < self.d + 1:
                        missing_point_index = self.d + 1 - len(x_sub)

                        x_sub = self.x_mesh[start - missing_point_index:end]

                        if self.y_values is None:
                            f_sub = f(x_sub)
                        else:
                            f_sub = self.y_values[start - missing_point_index:end]

                        div_coeff = copy.deepcopy(f_sub)
                        for i in range(1, len(x_sub)):
                            for j in range(len(x_sub) - 1, i - 1, -1):
                                div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (x_sub[j] - x_sub[j - i]))

                        self.div_coeff.append(div_coeff)
                        self.subintervals.append(x_sub)

                    else:
                        if self.y_values is None:
                            f_sub = f(x_sub)
                        else:
                            f_sub = self.y_values[start:end]

                        div_coeff = copy.deepcopy(f_sub)
                        for i in range(1, len(x_sub)):
                            for j in range(len(x_sub) - 1, i - 1, -1):
                                div_coeff[j] = self.dtype((div_coeff[j] - div_coeff[j - 1]) / (x_sub[j] - x_sub[j - i]))

                        self.div_coeff.append(div_coeff)
                        self.subintervals.append(x_sub)

                # print(f"Subinterval {s}: x_sub = {x_sub}")  # Debugging output
                # print(f"Divided Differences {s}: {div_coeff}\n")  # Debugging output

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

    def piecewise_interpolation(self, x, f, df, flag, hermite=False, specify=False):
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
            #s = 0
        elif x > mesh_max:
            # Option 1: Return None
            return None

            # Option 2: Use the last subinterval for extrapolation
            #s = self.M - 1
        else:
            # Find which subinterval contains x
            for s in range(self.M + 1):
                if not specify:
                    x_s = sorted_x_mesh[s * (self.d + 1)]
                    x_s_1 = sorted_x_mesh[(s + 1) * (self.d + 1)] if s < self.M - 1 else sorted_x_mesh[-1]
                else:
                    x_s = sorted_x_mesh[s]
                    x_s_1 = sorted_x_mesh[s + 1]

                if x_s <= x <= x_s_1:
                    break
            else:
                return None

        # Compute the Newton polynomial for this subinterval
        if not hermite:
            if not specify:
                idx_start = s * (self.d + 1)
                # x_submesh = sorted_x_mesh[idx_start: idx_start + self.d + 1]
                x_submesh = self.x_mesh[idx_start: idx_start + self.d + 1]
                coeffs = self.div_coeff[s]

                # Perform Newton's nested evaluation
                result = coeffs[-1]
                for i in range(self.d - 1, -1, -1):
                    result = result * (x - x_submesh[i]) + coeffs[i]
            else:
                x_submesh = sorted_x_mesh[s: s + self.d + 1]

                if len(x_submesh) < self.d + 1:
                    missing_point_index = self.d + 1 - len(x_submesh)

                    x_submesh = self.x_mesh[s - missing_point_index:s + self.d + 1]
                    coeffs = self.div_coeff[s]
                    result = coeffs[-1]
                    for i in range(self.d - 1, -1, -1):
                        result = result * (x - x_submesh[i]) + coeffs[i]

                else:
                    coeffs = self.div_coeff[s]

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

            third_divdiff = (f_b - f_a) / h ** 2 - df_a / h
            fourth_divdiff = (df_b + df_a) / h ** 2 - 2 * (f_b - f_a) / h ** 3

            result = f_a + df_a * (x - a_i) + third_divdiff * (x - a_i) ** 2 + fourth_divdiff * (x - a_i) ** 2 * (
                        x - b_i)

        return self.dtype(result)

    def tridiagonal_solve(self, lower, diag, upper, rhs, n):

        # Forward Substitution
        for i in range(1, n + 1):
            m = lower[i - 1] / diag[i - 1] # The multiplier being used
            diag[i] -= m * upper[i - 1] # Update the diagonal
            rhs[i] -= m * rhs[i - 1] # Update the right-hand-side

        # Back substitution
        result = np.zeros(n + 1, dtype=self.dtype)

        # Setting the last term
        result[n] = rhs[n] / diag[n]

        for i in range(n - 1, -1, -1):
            result[i] = (rhs[i] - upper[i] * result[i + 1]) / diag[i]

        return result

    def spline_interpolation(self, f, flag, second_deriv_0=None, second_deriv_n=None):

        if flag == 3:
            sorted_x_mesh = sorted(self.x_mesh)
        else:
            sorted_x_mesh = self.x_mesh

        n = len(sorted_x_mesh) - 1

        if self.y_values is None:
            y = f(sorted_x_mesh)
        else:
            y = self.y_values

        rhs = np.zeros(n + 1, dtype=self.dtype)
        lower = np.zeros(n, dtype=self.dtype)
        diag = np.zeros(n + 1, dtype=self.dtype)
        upper = np.zeros(n, dtype=self.dtype)

        for i in range(1, n):
            h_i = sorted_x_mesh[i] - sorted_x_mesh[i - 1]
            h_i_1 = sorted_x_mesh[i + 1] - sorted_x_mesh[i]

            lower[i - 1] = h_i
            diag[i] = 2 * (h_i + h_i_1)
            upper[i] = h_i_1

            rhs[i] = 3 * ((y[i + 1] - y[i]) / h_i_1 - (y[i] - y[i - 1]) / h_i)



        if second_deriv_0 is not None:
            diag[0] = 1
            rhs[0] = second_deriv_0 / 2
        else:
            diag[0] = 1
            rhs[0] = 0.0

        if second_deriv_n is not None:
            diag[n] = 1
            rhs[n] = second_deriv_n / 2
        else:
            diag[n] = 1
            rhs[n] = 0.0

        second_derivatives = self.tridiagonal_solve(lower, diag, upper, rhs, n)

        return self.dtype(second_derivatives)

    def get_spline_coefficients(self, second_derivatives, f, flag):
        if flag == 3:
            sorted_x_mesh = sorted(self.x_mesh)
        else:
            sorted_x_mesh = self.x_mesh

        n = len(sorted_x_mesh) - 1

        if self.y_values is None:
            y = f(sorted_x_mesh)
        else:
            y = self.y_values

        a = np.zeros(n, dtype=self.dtype)
        b = np.zeros(n, dtype=self.dtype)
        c = np.zeros(n, dtype=self.dtype)
        d = np.zeros(n, dtype=self.dtype)

        for i in range(n):
            h_i = sorted_x_mesh[i + 1] - sorted_x_mesh[i]

            a[i] = y[i]
            b[i] = (y[i + 1] - y[i]) / h_i - h_i * (2 * second_derivatives[i] + second_derivatives[i + 1]) / 3
            c[i] = second_derivatives[i]
            d[i] = (second_derivatives[i + 1] - second_derivatives[i]) / (3 * h_i)


        return a, b, c, d

    def evaluate_spline(self, x_point, a, b, c, d, flag):
        if flag == 3:
            sorted_x_mesh = sorted(self.x_mesh)
        else:
            sorted_x_mesh = self.x_mesh

        n = len(sorted_x_mesh) - 1

        i = 0
        while i < n and x_point > sorted_x_mesh[i + 1]:
            i += 1

        if i >= n:  # Handle case where x_point is at or beyond the right boundary
            i = n - 1

        diff_x = x_point - sorted_x_mesh[i]

        return d[i] * diff_x ** 3 + c[i] * diff_x ** 2 + b[i] * diff_x + a[i]

    def evaluate_spline_derivative(self, x_point, a, b, c, d, flag):
        if flag == 3:
            sorted_x_mesh = sorted(self.x_mesh)
        else:
            sorted_x_mesh = self.x_mesh

        n = len(sorted_x_mesh) - 1

        i = 0
        while i < n and x_point > sorted_x_mesh[i + 1]:
            i += 1

        diff_x = x_point - sorted_x_mesh[i]

        return b[i] + 2 * c[i] * diff_x + 3 * d[i] * diff_x **2

    def B_spline_interpolation(self, f, df):
        n = len(self.x_mesh) - 1
        h = np.diff(self.x_mesh)

        L = np.zeros((n + 3, n + 3))
        rhs = np.zeros(n + 3)

        rhs[0] = df(self.x_mesh[0])
        rhs[n + 2] = df(self.x_mesh[-1])

        for i in range(n + 1):
            rhs[i + 1] = f(self.x_mesh[i])



        for i in range(1, n + 2):
            L[i, i - 1] = 1
            L[i, i] = 4
            L[i, i + 1] = 1

        L[0, 0] = -3 / h[0]
        L[0, 1] = 0
        L[0, 2] = 3/ h[0]

        L[-1, -1] = 3 / h[-1]
        L[-1, -2] = 0
        L[-1, -3] = -3 / h[-1]

        alpha = np.linalg.solve(L, rhs)

        return alpha

    def B_spline_basis_K(self, x_point, i):

        if i < 0 or i >= len(self.x_mesh):
            return 0

        #h = np.diff(self.x_mesh)
        h = self.x_mesh[1] - self.x_mesh[0] if len(self.x_mesh) > 1 else 1

        # Handle extended mesh points
        def get_x(idx):
            if idx < 0:
                return self.x_mesh[0] - h * abs(idx)
            elif idx >= len(self.x_mesh):
                return self.x_mesh[-1] + h * (idx - len(self.x_mesh) + 1)
            return self.x_mesh[idx]

        x0 = get_x(i - 2)
        x1 = get_x(i - 1)
        x2 = get_x(i)
        x3 = get_x(i + 1)
        x4 = get_x(i + 2)

        if x_point < x0 or x_point > x4:
            return 0

        if x0 <= x_point <= x1:
            return (1 / h ** 3) * (x_point - x0)
        elif x1 <= x_point <= x2:
            return (1 / h ** 3) * (h ** 3 + 3 * h ** 2 * (x_point - x1) + 3 * h * (x_point - x1) ** 2 - 3 * (x_point - x1) ** 3)
        elif x2 <= x_point <= x3:
            return (1 / h ** 3) * (h ** 3 + 3 * h ** 2 * (x3 - x_point) + 3 * h * (x3 - x_point) ** 2 - 3 * (x3 - x_point) ** 3)
        elif x3 <= x_point <= x4:
            return (1 / h ** 3) * (x4 - x_point) ** 3


        return 0.0

    def evaluate_B_Spline(self, x_point, alpha):

        if x_point < self.x_mesh[0] or x_point > self.x_mesh[-1]:
            return 0.0

        
        i = 0
        while i < len(self.x_mesh) - 1 and x_point > self.x_mesh[i + 1]:
            i += 1

        result = 0

        for j in range(i-1, i+3):
            alpha_idx = j + 1

            if 0 <= alpha_idx < len(alpha):
                basis_val = self.B_spline_basis_K(x_point, j)
                result += alpha[alpha_idx] * basis_val

        return result







