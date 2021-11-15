import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from typing import List


def function1(x: float) -> float:
    return x ** 2


def function2(x: float) -> float:
    return 3 * x ** 3 - 1


def apply_function_list(n_func: int, x: List[float]) -> List[float]:
    if n_func == 1:
        return [function1(x_i) for x_i in x]
    elif n_func == 2:
        return [function2(x_i) for x_i in x]
    else:
        raise ValueError("Wrong function chosen")


def get_poly_value(coeffs: List[float], x: float) -> float:
    value = 0
    for i, coef in enumerate(coeffs):
        value += coef * (x ** i)
    return value


def get_basis_scalar_products(n: int, a: float, b: float) -> List[List[float]]:
    result_matrix = [[None for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            result_matrix[i][j] = (b ** (i + j + 1) - a ** (i + j + 1)) / (i + j + 1)
    return result_matrix


def get_function_basis_scalar_products(function_n: int, n: int, a: float, b: float) -> List[float]:
    result = []
    if function_n == 1:
        f = function1
    elif function_n == 2:
        f = function2
    else:
        raise ValueError("Function does not exist")

    for i in range(n + 1):
        f_integrate = lambda x: f(x) * (x ** i)
        result.append(integrate.quad(f_integrate, a, b)[0])
    return result


def gauss_solve(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = - a[i][k] / a[k][k]
            b[i] += m * b[k]
            for j in range(k + 1, n):
                a[i][j] += m * a[k][j]
    x = [None] * n
    x[n - 1] = b[n - 1] / a[n - 1][n - 1]
    for k in range(n - 2, -1, -1):
        temp_sum = 0
        for j in range(k + 1, n):
            temp_sum += a[k][j] * x[j]
        x[k] = (b[k] - temp_sum) / a[k][k]
    return x


def best_approximation(n: int, a: float, b: float, function_n: int, q=500):
    matrix = get_basis_scalar_products(n, a, b)
    vector = get_function_basis_scalar_products(function_n, n, a, b)
    coeffs = gauss_solve(matrix, vector)
    x_plot = np.linspace(a, b, num=q)
    y_func = apply_function_list(function_n, x_plot)
    y_poly = [get_poly_value(coeffs, point) for point in x_plot]

    plt.plot(x_plot, y_func, 'k-', label="f(x)")
    plt.plot(x_plot, y_poly, 'r-', label=f"$p_{n}(x)$")
    plt.legend(loc='best')
    plt.show()
