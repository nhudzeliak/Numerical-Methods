import numpy as np
from typing import Tuple, List


def read_file(filename: str) -> Tuple[List[float], List[float]]:
    a = []
    b = []
    with open(filename, "r") as f:
        for row in f:
            r, l = row.strip().split("|")
            a.append(list(map(float, r.split())))
            b.append(int(l))
    return a, b


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


def solve_system(filename: str) -> Tuple[List[float], List[float], float]:
    a, b = read_file(filename)
    m = len(a)
    n = len(a[0])
    a = np.array(a)
    b = np.array(b).reshape(m, 1)
    a_square = a.T @ a
    vect = a.T @ b
    x_approxim = gauss_solve(a_square.tolist(), vect.reshape(1, -1)[0].tolist())
    error = a @ np.array(x_approxim).reshape(n, 1) - b
    error_norm = np.max(np.absolute(error))
    return x_approxim, error.reshape(1, -1)[0].tolist(), error_norm
