from typing import List, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import math


def function(x: float) -> float:
    return math.exp(math.sin(x) + math.cos(x))


def f_derivative(x: float) -> float:
    return math.exp(math.sin(x) + math.cos(x)) * (math.cos(x) - math.sin(x))


def get_x(a: float, b: float, n: int) -> Tuple[List[float], float]:
    h = (b - a) / n
    x = [a + k * h for k in range(n + 1)]
    return x, h


def get_y(x: Union[List[float], np.ndarray]) -> List[float]:
    return [function(element) for element in x]


def b1(x: float) -> float:
    if abs(x) <= 1:
        return 1 - abs(x)
    else:
        return 0


def b3(x: float) -> float:
    if abs(x) <= 1:
        return (1 / 6) * ((2 - abs(x)) ** 3 - 4 * ((1 - abs(x)) ** 3))
    elif 1 <= abs(x) <= 2:
        return (1 / 6) * ((2 - abs(x)) ** 3)
    else:
        return 0


def solve_system(matrix: List[List[float]], f: List[float]) -> List[float]:
    n = len(matrix) - 1
    c = [-matrix[i][i] for i in range(n + 1)]
    a = [None]
    b = []
    for i in range(n + 1):
        for j in range(n + 1):
            if j - i == 1:
                b.append(matrix[i][j])
            if i - j == 1:
                a.append(matrix[i][j])
    xi1, xi2 = -b[0], -a[n]
    mu1, mu2 = f[0], f[-1]
    alpha, beta = [None, xi1], [None, mu1]
    for i in range(1, n):
        alpha.append(b[i] / (c[i] - a[i] * alpha[i]))
        beta.append((-f[i] + a[i] * beta[i]) / (c[i] - a[i] * alpha[i]))
    y = [None] * (n + 1)
    y[n] = (mu2 + xi2 * beta[n]) / (1 - xi2 * alpha[n])
    for i in range(n-1, -1, -1):
        y[i] = alpha[i + 1] * y[i + 1] + beta[i + 1]
    return y


def linear_spline(x: List[float], y: List[float], n: int, h: float, val: float) -> float:
    result = 0
    for i in range(n + 1):
        result += y[i] * b1((val - x[i]) / h)
    return result


def cubic_spline(a: float, b: float, n: int, val: float) -> float:
    x, h = get_x(a, b, n)
    y = get_y(x)
    a1 = f_derivative(a)
    b1_ = f_derivative(b)
    f = [3 * y[0] - h * a1]
    for i in range(n + 1):
        f.append(y[i])
    f.append(3 * y[n] + h * b1_)
    matr = [[0 for _ in range(n + 3)] for _ in range(n + 3)]
    matr[0][0] = matr[n + 2][n + 2] = 1
    matr[0][1] = matr[n + 2][n + 1] = 2
    for i in range(1, n + 2):
        for j in range(n + 3):
            if j == i:
                matr[i][j] = 2 / 3
            if abs(i - j) == 1:
                matr[i][j] = 1 / 6
    alpha = solve_system(matr, f)
    x.insert(0, a - h)
    x.append(b + h)
    result = 0
    for i in range(n + 3):
        result += alpha[i] * b3((val - x[i]) / h)
    return result


def spline_interpolation(a: float, b: float, n: int, q: int) -> None:
    x_nodes, h = get_x(a, b, n)
    y_nodes = get_y(x_nodes)
    x_plot = np.linspace(a, b, num=q)
    y_func = get_y(x_plot)
    y_s1 = [linear_spline(x_nodes, y_nodes, n, h, val) for val in x_plot]
    y_s3 = [cubic_spline(a, b, n, val) for val in x_plot]

    plt.plot(x_plot, y_func, 'k-', label="f(x)")
    plt.plot(x_plot, y_s1, 'r-', label="s1(x)")
    plt.plot(x_plot, y_s3, 'b-', label="s3(x)")
    plt.scatter(x_nodes, y_nodes, color='green', label="nodes")
    plt.legend(loc='best')
    plt.show()


