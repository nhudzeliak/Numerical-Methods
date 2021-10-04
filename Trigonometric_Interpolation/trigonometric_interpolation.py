from typing import List
from math import pi, sin, cos, exp, fabs
import numpy as np
import matplotlib.pyplot as plt


def function1(x: float) -> float:
    return exp(sin(x) + cos(x))


def function2(x: float) -> float:
    return 3 * cos(15 * x)


def apply_function_list(n_func: int, x: List[float]) -> List[float]:
    if n_func == 1:
        return [function1(x_i) for x_i in x]
    elif n_func == 2:
        return [function2(x_i) for x_i in x]
    else:
        raise ValueError("Wrong function chosen")


def apply_function_scalar(n_func: int, x: float) -> float:
    if n_func == 1:
        return function1(x)
    elif n_func == 2:
        return function2(x)
    else:
        raise ValueError("Wrong function chosen")


def get_a_coefficient(k: int, n: int, y: List[float]) -> float:
    """ Evaluation of a k-th """
    summation = 0
    for j in range(2 * n):
        summation += y[j] * cos((pi * j * k) / n)  # cos(x[j] * k)
    return summation / n


def get_b_coefficient(k: int, n: int, y: List[float]) -> float:
    """ Evaluation of b k-th """
    summation = 0
    for j in range(2 * n):
        summation += y[j] * sin((pi * j * k) / n)  # sin(x[j] * k)
    return summation / n


def polynomial_value(a: List[float], b: List[float], n: int, x: float) -> float:
    summation = 0
    for k in range(1, n):
        summation += a[k] * cos(k * x) + b[k] * sin(k * x)
    return a[0] / 2 + summation + (a[n] / 2) * cos(n * x)


def get_nodes(n: int) -> List[float]:
    return [(pi * j) / n for j in range(2 * n)]


def interpolate(n_func: int, n: int, x_error: float, m: int = 1000) -> None:
    x = get_nodes(n)
    y = apply_function_list(n_func, x)
    a = [get_a_coefficient(k, n, y) for k in range(n + 1)]
    b = [None]
    b += [get_b_coefficient(k, n, y) for k in range(1, n)]

    error = fabs(apply_function_scalar(n_func, x_error) - polynomial_value(a, b, n, x_error))
    print(f"Error at x={x_error}: {error}")

    # Visualization
    x_to_plot = np.linspace(0, 2 * pi, num=m)
    y_func = np.array(apply_function_list(n_func, x_to_plot))
    y_poly = np.array([polynomial_value(a, b, n, x) for x in x_to_plot])

    plt.plot(x_to_plot, y_func, 'k-', label="f(x)")
    plt.plot(x_to_plot, y_poly, 'r-', label=f"$T_{n}(x)$")

    plt.scatter(np.array(x), np.array(y), color='b', label='nodes')

    plt.legend(loc='best')
    plt.show()
