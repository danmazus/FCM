import numpy as np

class NumericalQuadrature:
    def __init__(self, a, b, m, f):
        self.a = a  # Left-endpoint of global interval
        self.b = b  # Right-endpoint of global interval
        self.m = m  # Number of subintervals to be taken

        # Initialize H_m
        self.H_m = (self.b - self.a) / self.m

        # Initialize the global mesh which splits the global interval of integration into m subintervals
        self.global_mesh = np.linspace(self.a, self.b, self.m + 1)
        self.x_mesh = None

        # Set the function to be used and the 2nd derivative if given
        self.f = f
        self.f_double_prime = None



    def composite_midpoint(self) -> float:
        # Defining the local step size
        h_mp = self.H_m / 2

        # Creating the mesh
        self.x_mesh = np.array([self.global_mesh[i] + h_mp for i in range(self.m)])

        # Computing the approximation
        I_m = self.H_m * sum(self.f(self.x_mesh))


        return I_m

    def composite_2_point(self) -> float:
        # Defining the local step size
        h_2p = self.H_m / 3

        # Creating the mesh
        self.x_mesh = np.array([[self.global_mesh[i] + j * h_2p for j in range(1, 3)] for i in range(self.m)]).flatten()

        # Computing the approximation
        I_m = (self.H_m / 2) * sum(self.f(self.x_mesh))

        return I_m

    def composite_trapezoid(self, adaptive=False, tol=1e-6) -> float | None:
        # Choosing whether adaptive is being used or not (default of not)
        if not adaptive:
            # Create the mesh (don't need local step size)
            self.x_mesh = np.array([self.global_mesh[0] + i * self.H_m for i in range(self.m + 1)])

            # Compute the approximation
            I_m = (self.H_m / 2) * (self.f(self.x_mesh[0]) + self.f(self.x_mesh[-1]) + 2 * sum(self.f(self.x_mesh[1:self.m])))

            return I_m
        # else:
        #     alpha = 1 / 2
        #     I_old = self.composite_trapezoid(adaptive=False)
        #
        #     if I_old < tol:
        #         return I_old
        #
        #     m = self.m
        #     H_m = self.H_m
        #
        #
        #
        #
        #     return I_m
        return None

    def composite_simpson_first(self) -> float:
        # Define the local step size
        h_sf = self.H_m / 2

        # Create the mesh
        self.x_mesh = np.array([self.global_mesh[i] + h_sf for i in range(self.m)])

        # Compute the approximation
        I_m = (self.H_m / 6) * (self.f(self.global_mesh[0]) + self.f(self.global_mesh[-1]) + 2 * sum(self.f(self.global_mesh[1:self.m])) + 4 * sum(self.f(self.x_mesh)))

        return I_m

    def composite_left_rectangle(self) -> float:
        # Compute the approximation (don't need local mesh as we take the left endpoints of the subintervals
        I_m = (self.H_m) * sum((self.f(self.global_mesh[0:self.m])))

        return I_m

    def gauss_2_point_quadrature(self) -> float:
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


def f(x):
    #return np.exp(x)
    return np.exp(np.sin(2 * x)) * np.cos(2 * x)
    #return np.power(x, 3) + np.power(x, 2) + x + 1

def df(x):
    return np.exp(x)

def dff(x):
    return np.exp(x)

a = 0
b = (np.pi/3)
m = 10

y_true = (1/2)*(-1 + np.exp(np.sqrt(3)/2))
print(y_true)
quad_midpoint = NumericalQuadrature(a, b, m, f)
result = quad_midpoint.composite_midpoint()
print(result)


quad_2_point = NumericalQuadrature(a, b, m, f)
result = quad_2_point.composite_2_point()
print(result)


quad_trapezoid = NumericalQuadrature(a, b, m, f)
result = quad_trapezoid.composite_trapezoid()
print(result)


quad_simpson_first = NumericalQuadrature(a, b, m, f)
result = quad_simpson_first.composite_simpson_first()
print(result)


quad_left_rectangle = NumericalQuadrature(a, b, m, f)
result = quad_left_rectangle.composite_left_rectangle()
print(result)

quad_gauss = NumericalQuadrature(a, b, m, f)
result = quad_gauss.gauss_2_point_quadrature()
print(result)
