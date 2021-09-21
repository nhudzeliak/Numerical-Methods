from typing import List, Optional, Union
import math
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy as np


def evenly_spaced_nodes(a: float, b: float, n: int) -> List[float]:
    step = (b - a) / n
    return [a + k * step for k in range(n + 1)]


def chebyshev_nodes(a: float, b: float, n: int) -> List[float]:
    return sorted([(b + a) / 2 + ((b - a) / 2) * math.cos((2 * k + 1) * math.pi / (2 * (n + 1)))
                   for k in range(n + 1)])


def function1(x: float) -> float:
    return 1 / (1 + 25 * (x ** 2))


def function2(x: float) -> float:
    return math.log(x + 2)


def function3(x: float) -> float:
    return x ** 3 - 2 * x + 6


def apply_function(n_func: int, x: Union[List[float], np.ndarray]) -> List[float]:
    if n_func == 1:
        return [function1(x_i) for x_i in x]
    elif n_func == 2:
        return [function2(x_i) for x_i in x]
    elif n_func == 3:
        return [function3(x_i) for x_i in x]
    else:
        raise ValueError("Wrong function chosen")


def apply_function_to_scalar(n_func, x: float) -> float:
    if n_func == 1:
        return function1(x)
    elif n_func == 2:
        return function2(x)
    elif n_func == 3:
        return function3(x)
    else:
        raise ValueError("Wrong function chosen")


def lagrange(x: List[float], y: List[float]) -> np.ndarray:
    n = len(x) - 1
    p = 0
    for i in range(n + 1):
        numerator = 1
        denominator = 1
        for k in range(n + 1):
            if k != i:
                numerator = poly.polymul(numerator, (-x[k], 1))
                denominator *= x[i] - x[k]
        l = numerator / denominator
        p = poly.polyadd(p, y[i] * l)
    return p


def get_divided_differences(x: List[float], y: List[float]) -> List[List[Optional[float]]]:
    n = len(x) - 1
    d = [[None for i in range(n + 1)] for j in range(n + 1)]
    for i in range(n + 1):
        d[0][i] = y[i]
    for k in range(1, n+1):
        for j in range(n - k + 1):
            d[k][j] = (d[k - 1][j + 1] - d[k - 1][j]) / (x[j + k] - x[j])
    return d


def newton(x: List[float], y: List[float]) -> np.ndarray:
    n = len(x) - 1
    d = get_divided_differences(x, y)
    p = 0
    for i in range(n + 1):
        multiplier = 1
        for k in range(i):
            multiplier = poly.polymul(multiplier, (-x[k], 1))
        p = poly.polyadd(p, d[i][0] * multiplier)
    return p


def interpolate(n_func: int, a: float, b: float, n: int, x: float, m: int) -> None:
    x_evenly = evenly_spaced_nodes(a, b, n)
    y_evenly = apply_function(n_func, x_evenly)
    p_lagrange = lagrange(x_evenly, y_evenly)
    p_newton = newton(x_evenly, y_evenly)

    x_chebyshev = chebyshev_nodes(a, b, n)
    y_chebyshev = apply_function(n_func, x_chebyshev)
    p_lagrange_chebyshev = lagrange(x_chebyshev, y_chebyshev)
    p_newton_chebyshev = newton(x_chebyshev, y_chebyshev)

    error = math.fabs(apply_function_to_scalar(n_func, x) - poly.polyval(x, p_lagrange))
    error_chebyshev = math.fabs(apply_function_to_scalar(n_func, x) - poly.polyval(x, p_lagrange_chebyshev))

    np.polynomial.set_default_printstyle('unicode')
    print("Results:")
    print("1) Evenly spaced nodes:")
    print(f"\tLagrange polynomial: {poly.Polynomial(p_lagrange)}")
    print(f"\tNewton polynomial: {poly.Polynomial(p_newton)}")
    print(f"\tPolynomial value at x={x}: {poly.polyval(x, p_lagrange)}")
    print(f"\tError: {error}")
    print("2) Chebyshev nodes:")
    print(f"\tLagrange polynomial: {poly.Polynomial(p_lagrange_chebyshev)}")
    print(f"\tNewton polynomial: {poly.Polynomial(p_newton_chebyshev)}")
    print(f"\tPolynomial value at x={x}: {poly.polyval(x, p_lagrange_chebyshev)}")
    print(f"\tError: {error_chebyshev}")
    # Visualization
    x_to_plot = np.linspace(a, b, num=m)
    y_func = np.array(apply_function(n_func, x_to_plot))
    y_ev = poly.polyval(x_to_plot, p_lagrange)
    y_ch = poly.polyval(x_to_plot, p_lagrange_chebyshev)
    plt.plot(x_to_plot, y_func, 'k-', label="f(x)")
    plt.plot(x_to_plot, y_ev, 'b-', label=f"$p_{n}(x)$(evenly spaced nodes)")
    plt.plot(x_to_plot, y_ch, 'r-', label=f"$p_{n}(x)$(chebyshev nodes)")
    plt.legend(loc='best')
    plt.show()
