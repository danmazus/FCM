import numpy as np
import copy
from my_package import *

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
                        0.5 * (self.b - self.a) * np.cos(((2 * i + 1) * np.pi) / (2 * self.n + 2)) + 0.5 * (
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

    def newton_divdiff(self, f, piecewise=False):
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
            # s = self.M - 1
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
            # x_submesh = sorted_x_mesh[idx_start: idx_start + self.d + 1]
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

            third_divdiff = (f_b - f_a) / h ** 2 - df_a / h
            fourth_divdiff = (df_b + df_a) / h ** 2 - 2 * (f_b - f_a) / h ** 3

            result = f_a + df_a * (x - a_i) + third_divdiff * (x - a_i) ** 2 + fourth_divdiff * (x - a_i) ** 2 * (
                        x - b_i)

        return self.dtype(result)

    def spline_interpolation(self, f, df, degree):
        if degree not in ['quadratic', 'cubic']:
            raise ValueError('Degree must be one of: "quadratic", "cubic"')

        n = len(self.x_mesh) - 1

        y = f(self.x_mesh)
        dy = df(self.x_mesh)

        rhs = np.zeros(n + 1, dtype=self.dtype)

        if degree == 'quadratic':
            rhs[0] = 2 * dy[0]
            rhs[n] = 2 * dy[n]
        elif degree == 'cubic':
            rhs[0] = 0
            rhs[n] = 0

        L = np.zeros((n + 1, n + 1), dtype=self.dtype)

        if degree == 'quadratic':
            L[0, 0] = 1
            L[n, n] = 1
            for i in range(1, n):
                h_i = self.x_mesh[i] - self.x_mesh[i - 1]

                L[i, i] = 1

                L[i, i - 1] = 1

                rhs[i] = 2 * (y[i] - y[i-1]) / h_i

            #first_derivatives = solve_Lb_np(L, rhs)
            first_derivatives = np.linalg.solve(L, rhs)

            return self.dtype(first_derivatives)

        elif degree == 'cubic':
            L[0, 0] = 1
            L[n, n] = 1

            for i in range(1, n):
                h_i = self.x_mesh[i] - self.x_mesh[i - 1]
                h_next = self.x_mesh[i + 1] - self.x_mesh[i]

                L[i, i] = 2
                L[i, i - 1] = h_i
                L[i, i + 1] = h_next

                rhs[i] = 6 * ((y[i + 1] - y[i]) / h_next - (y[i] - y[i - 1]) / h_i)

            second_derivatives = np.linalg.solve(L, rhs)

            return self.dtype(second_derivatives)


    def get_spline_coefficients(self, derivatives, f, degree):
        n = len(self.x_mesh) - 1

        y = f(self.x_mesh)

        a = np.zeros(n + 1, dtype=self.dtype)
        b = np.zeros(n + 1, dtype=self.dtype)
        c = np.zeros(n + 1, dtype=self.dtype)
        d = np.zeros(n, dtype=self.dtype)

        if degree == 'quadratic':
            for i in range(n):
                h_i = self.x_mesh[i + 1] - self.x_mesh[i]

                c[i] = y[i]
                b[i] = derivatives[i]
                a[i] = (y[i+1] - y[i] - b[i] * h_i) / h_i ** 2


            return a, b, c

        # elif degree == 'cubic':
        #     for i in range(n):
        #         h_i = self.x_mesh[i + 1] - self.x_mesh[i]
        #
        #         c[i] = second_derivatives[i]
        #         b[i] = derivatives[i]
        #         a[i] =



    def evaluate_spline(self, x_point, a, b, c):
        n = len(self.x_mesh) - 1

        i = 0
        while i < n and x_point > self.x_mesh[i + 1]:
            i += 1

        diff_x = x_point - self.x_mesh[i]

        return a[i] * diff_x ** 2 + b[i] * diff_x + c[i]



