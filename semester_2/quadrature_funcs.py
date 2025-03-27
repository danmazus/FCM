import numpy as np

class NumericalQuadrature:
    def __init__(self, a, b, n, f, open=False):
        self.a = a  # Left-endpoint of global interval
        self.b = b  # Right-endpoint of global interval
        self.n = n  # Number of points specified
        if open:
            self.h_n = (self.b - self.a) / (self.n + 2)
            self.x_mesh = np.array([self.a + (i + 1) * self.h_n for i in range(self.n + 1)])
        else:
            self.h_n = (self.b - self.a) / (self.n + 1)
            self.x_mesh = np.linspace(self.a, self.b, self.n + 1)


        self.f = f
        self.f_double_prime = None



    def composite_midpoint(self) -> float:
        I_m = 2 * self.h_n * self.f(self.x_mesh[0])
        error = None

        return I_m

    def composite_2_point(self) -> float:
        I_m = 3 * self.h_n * (self.f(self.x_mesh[0]) + self.f(self.x_mesh[1]))

        return I_m

    def composite_trapezoid(self) -> float:
        I_m = (self.h_n / 2) * (self.f(self.x_mesh[0]) + self.f(self.x_mesh[1]))

        return I_m

    def composite_simpson_first(self) -> float:
        I_m = (self.h_n / 3) * (self.f(self.x_mesh[0]) + 4 * self.f(self.x_mesh[1]) + self.f(self.x_mesh[2]))

        return I_m

    def composite_left_rectangle(self) -> float:
        I_m = (self.h_n) * (self.f(self.x_mesh[0]))

        return I_m


def f(x):
    return np.exp(x)

def df(x):
    return np.exp(x)

def dff(x):
    return np.exp(x)

a = 0
b = 1

y_true = np.exp(1)
quad_midpoint = NumericalQuadrature(a, b, 0, f, open=True)
print(quad_midpoint.x_mesh)
result = quad_midpoint.composite_midpoint()
print(result)


quad_2_point = NumericalQuadrature(a, b, 1, f, open=True)
print(quad_2_point.x_mesh)
result = quad_2_point.composite_2_point()
print(result)

quad_trapezoid = NumericalQuadrature(a, b, 1, f, open=False)
print(quad_trapezoid.x_mesh)
result = quad_trapezoid.composite_trapezoid()
print(result)

quad_simpson_first = NumericalQuadrature(a, b, 2, f, open=False)
print(quad_simpson_first.x_mesh)
result = quad_simpson_first.composite_simpson_first()
print(result)

quad_left_rectangle = NumericalQuadrature(a, b, 0, f, open=False)
print(quad_left_rectangle.x_mesh)
result = quad_left_rectangle.composite_left_rectangle()
print(result)