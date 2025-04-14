import numpy as np
import math
import matplotlib.pyplot as plt

class NumericalQuadrature:
    def __init__(self, a, b, m, f, epsilon=None, f_deriv_max=None):
        self.a = a  # Left-endpoint of global interval
        self.b = b  # Right-endpoint of global interval
        self.m = m  # Number of subintervals to be taken

        self.epsilon = epsilon
        self.f_deriv_max = f_deriv_max

        # Initialize the global mesh which splits the global interval of integration into m subintervals
        self.x_mesh = None

        # Set the function to be used and the 2nd derivative if given
        self.f = f
        self.f_double_prime = None

    def compute_H_m(self, method: str):
        """
        Computes the required subinterval size H_m for a desired error tolerance level, denoted as epsilon. This function
        uses different formulas based on the method given (i.e. Composite Trapezoidal, Composite Midpoint, etc.)

        Inputs:
            - epsilon: Desired Error Tolerance Level
            - f_deriv_max: Maximum value of the required derivative for the specific method
                - 2nd derivative for Composite Midpoint and Composite Trapezoidal
                - 4th Derivative for Composite Simpson's Method
            - method: Quadrature method to be used
                - 'trapezoidal': Composite Trapezoidal method
                - 'midpoint': Composite Midpoint method
                - 'simpson': Composite Simpson's First method
                - 'left_rectangle': Composite Left Rectangle method
                - 'gauss_leg_two_point': Composite Gauss-Legendre Two Point method
                - 'two_point': Composite Two Point method

        Outputs:
            - H_m: Required subinterval size to achieve the desired error tolerance level
        """

        if method == 'trapezoidal':
            self.H_m = np.sqrt((12 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max))
        elif method == 'midpoint':
            self.H_m = np.sqrt((24 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max))
        elif method == 'simpson':
            self.H_m = ((2880 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max)) ** 0.25
        elif method == 'left_rectangle':
            self.H_m = ((2 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max))
        elif method == 'gauss_leg_two_point':
            self.H_m = ((4320 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max)) ** 0.25
        elif method == 'two_point':
            self.H_m = np.sqrt((36 * self.epsilon) / ((self.b - self.a) * self.f_deriv_max))
        else:
            raise ValueError("Method is not supported and defined. Use 'trapezoidal', 'midpoint', 'simpson', "
                             "'left_rectangle', 'gauss_leg_two_point', 'two_point'")

        return self.H_m

    def compute_m(self):
        if self.H_m is None:
            raise ValueError("H_m must be computed before computing m.")

        self.m = math.ceil((self.b - self.a) / self.H_m)

        return self.m

    def composite_midpoint(self, optimal_H_m=False, adaptive=False, tol=None, max_iter=2000, y_true=None) -> float:
        if not adaptive:
            if optimal_H_m:
                if self.epsilon is None or self.f_deriv_max is None:
                    raise ValueError("Both epsilon and f_deriv_max must be defined.")

                self.compute_H_m(method='midpoint')

                self.compute_m()

            self.H_m = (self.b - self.a) / self.m

            # Define global mesh
            self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

            # Defining the local step size
            h_mp = self.H_m / 2

            # Creating the mesh
            self.x_mesh = np.array([self.global_mesh[i] + h_mp for i in range(self.m)])

            # Computing the approximation
            I_m = self.H_m * np.sum(self.f(self.x_mesh))


            return I_m

        else:
            alpha = 1 / 3
            I_old = self.composite_midpoint(adaptive=False)

            H_m = self.H_m
            iter_count = 0

            while iter_count < max_iter:
                old_H_m = H_m
                H_m *= alpha

                f_new_sum = 0

                x_mid = self.a + (H_m / 2)
                while x_mid < self.b:
                    f_new_sum += self.f(x_mid)
                    x_mid += old_H_m

                x_mid = self.a + ((5 * H_m) / 2)
                while x_mid < self.b:
                    f_new_sum += self.f(x_mid)
                    x_mid += old_H_m

                I_new = (1 / 3) * I_old + H_m * f_new_sum

                r = 2

                error_estimate = abs((I_new - I_old) / (2 ** r - 1))

                if y_true is not None:
                    if abs(y_true - I_new) < tol:
                        self.m = (self.b - self.a) / H_m
                        self.H_m = H_m
                        return I_new
                else:
                    if error_estimate < tol:
                        self.m = (self.b - self.a) / H_m
                        self.H_m = H_m
                        return I_new

                I_old = I_new
                iter_count += 1

            return I_old

    def composite_2_point(self, optimal_H_m=False) -> float:
        if optimal_H_m:
            if self.epsilon is None or self.f_deriv_max is None:
                raise ValueError("Both epsilon and f_deriv_max must be defined.")

            self.compute_H_m(method='two_point')

            self.compute_m()

        self.H_m = (self.b - self.a) / self.m


        self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

        # Defining the local step size
        h_2p = self.H_m / 3

        # Creating the mesh
        self.x_mesh = np.array([[self.global_mesh[i] + j * h_2p for j in range(1, 3)] for i in range(self.m)]).flatten()

        # Computing the approximation
        I_m = (self.H_m / 2) * np.sum(self.f(self.x_mesh))

        return I_m

    def composite_trapezoid(self, optimal_H_m=False, adaptive=False, tol=None, max_iter=2000, y_true=None) -> float | None:
        # Choosing whether adaptive is being used or not (default of not)
        if not adaptive:
            if optimal_H_m:
                if self.epsilon is None or self.f_deriv_max is None:
                    raise ValueError("Both epsilon and f_deriv_max must be defined.")

                # Compute the optimal H_m
                self.compute_H_m(method='trapezoidal')

                # Compute m subintervals from H_m
                self.compute_m()

            self.H_m = (self.b - self.a) / self.m

            # Split the interval into m subintervals
            self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

            # Compute the approximation
            I_m = (self.H_m / 2) * (self.f(self.global_mesh[0]) + self.f(self.global_mesh[-1]) +
                                    2 * np.sum(self.f(self.global_mesh[1:-1])))

            return I_m
        else:
            alpha = 1 / 2
            I_old = self.composite_trapezoid(adaptive=False)

            if abs(y_true - I_old) < tol:
                return I_old

            H_m = self.H_m
            iter_count = 0

            while iter_count < max_iter:
                H_m *= alpha

                x_mid = self.a + H_m
                f_new_sum = 0

                while x_mid < self.b:
                    f_new_sum += self.f(x_mid)
                    x_mid += 2 * H_m

                r = 2

                I_new = 0.5 * I_old + H_m * f_new_sum

                error_estimate = abs(I_new - I_old) / (2 ** r - 1)

                if y_true is not None:
                    if abs(y_true - I_new) < tol:
                        self.m = (self.b - self.a) / H_m
                        self.H_m = H_m
                        return I_new
                else:
                    if error_estimate < tol:
                        self.m = (self.b - self.a) / H_m
                        self.H_m = H_m
                        return I_new


                I_old = I_new
                iter_count += 1

            return I_old

    def composite_simpson_first(self, optimal_H_m=False) -> float:
        if optimal_H_m:
            if self.epsilon is None or self.f_deriv_max is None:
                raise ValueError("Both epsilon and f_deriv_max must be defined.")

            # Compute the optimal H_m
            self.compute_H_m(method='simpson')

            # Compute number of subintervals m from H_m
            self.compute_m()


        self.H_m = (self.b - self.a) / self.m


        # Split the given interval [a,b] into m subintervals
        self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

        # Define the local step size
        h_sf = self.H_m / 2

        # Create the mesh
        self.x_mesh = np.array([self.global_mesh[i] + h_sf for i in range(self.m)])

        # Compute the approximation
        I_m = (self.H_m / 6) * (self.f(self.global_mesh[0]) + self.f(self.global_mesh[-1]) + 2 *
                                np.sum(self.f(self.global_mesh[1:self.m])) + 4 * np.sum(self.f(self.x_mesh)))

        return I_m

    def composite_left_rectangle(self, optimal_H_m=False) -> float:
        if optimal_H_m:
            if self.epsilon is None or self.f_deriv_max is None:
                raise ValueError("Both epsilon and f_deriv_max must be defined.")

            self.compute_H_m(method='left_rectangle')

            self.compute_m()


        self.H_m = (self.b - self.a) / self.m

        self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

        # Compute the approximation (don't need local mesh as we take the left endpoints of the subintervals
        I_m = (self.H_m) * np.sum((self.f(self.global_mesh[0:self.m])))

        return I_m

    def gauss_2_point_quadrature(self, optimal_H_m=False) -> float:
        if optimal_H_m:
            if self.epsilon is None or self.f_deriv_max is None:
                raise ValueError("Both epsilon and f_deriv_max must be defined.")

            self.compute_H_m(method='gauss_leg_two_point')

            self.compute_m()


        self.H_m = (self.b - self.a) / self.m

        self.global_mesh = np.linspace(self.a, self.b, self.m + 1)

        # Define the positive x
        x_pos = 1 / (np.sqrt(3))

        # Define the negative x
        x_neg = -(1 / np.sqrt(3))

        # Initialize the approximation
        I_m = 0

        # Loop over the m-subintervals
        for i in range(self.m):
            # Compute the x_pos from the reference interval to the interval of interpolation
            x1 = (((self.global_mesh[i + 1] - self.global_mesh[i]) / 2) * x_pos) + (self.global_mesh[i+1] + self.global_mesh[i]) / 2

            # Compute the x_neg from the reference interval to the interval of interpolation
            x2 = (((self.global_mesh[i + 1] - self.global_mesh[i]) / 2) * x_neg) + (self.global_mesh[i+1] + self.global_mesh[i]) / 2

            # Compute the approximation
            I_m += ((self.global_mesh[i + 1] - self.global_mesh[i]) / 2) * (self.f(x1) + self.f(x2))

        return I_m