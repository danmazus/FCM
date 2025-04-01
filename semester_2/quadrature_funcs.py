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

        h_mp = self.H_m / 2
        self.x_mesh = np.array([self.global_mesh[i] + h_mp for i in range(self.m)])
        I_m = self.H_m * sum(self.f(self.x_mesh))


        return I_m

    def composite_2_point(self) -> float:
        h_2p = self.H_m / 3

        # self.x_mesh = np.array([self.global_mesh[i] + h_2p for i in range(self.m)])
        # I_m = h_2p * sum((self.f(self.x_mesh[0:self.m]) + self.f(self.x_mesh[1:self.m+1])))

        self.x_mesh = np.array([[self.global_mesh[i] + j * h_2p for j in range(1, 3)] for i in range(self.m)]).flatten()
        I_m = (self.H_m / 2) * sum(self.f(self.x_mesh))

        return I_m

    def composite_trapezoid(self, adaptive=False) -> float | None:
        if not adaptive:
            self.x_mesh = np.array([self.global_mesh[0] + i * self.H_m for i in range(self.m + 1)])
            I_m = (self.H_m / 2) * (self.f(self.x_mesh[0]) + self.f(self.x_mesh[-1]) + 2 * sum(self.f(self.x_mesh[1:self.m])))

            return I_m
        # else:
        #     alpha = 1 / 2
        #
        #     return I_m
        return None

    def composite_simpson_first(self) -> float:
        h_sf = self.H_m / 2
        self.x_mesh = np.array([self.global_mesh[i] + h_sf for i in range(self.m)])

        I_m = (self.H_m / 6) * (self.f(self.global_mesh[0]) + self.f(self.global_mesh[-1]) + 2 * sum(self.f(self.global_mesh[1:self.m])) + 4 * sum(self.f(self.x_mesh)))

        return I_m

    def composite_left_rectangle(self) -> float:
        I_m = (self.H_m) * sum((self.f(self.global_mesh[0:self.m])))

        return I_m

    def gauss_2_point_quadrature(self) -> float:
        x_pos = 1 / (np.sqrt(3))
        x_neg = -(1 / np.sqrt(3))
        I_m = 0
        for i in range(self.m):
            x1 = (((self.global_mesh[i + 1] - self.global_mesh[i]) / 2) * x_pos) + (self.global_mesh[i+1] + self.global_mesh[i]) / 2
            x2 = (((self.global_mesh[i + 1] - self.global_mesh[i]) / 2) * x_neg) + (self.global_mesh[i+1] + self.global_mesh[i]) / 2
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
m = 3

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
