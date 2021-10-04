from math import pi
from trigonometric_interpolation import interpolate


def main():
    n_func = int(input("Choose function:\n1 - e^(sin(x) + cos(x))\n2 - 3cos(15x) "))
    n = int(input("Enter n: "))
    while n <= 0:
        n = int(input("Enter valid n: "))
    x = float(input("Enter xє[0,2pi]: "))
    while x < 0 or x > 2 * pi:
        x = float(input("Enter xє[0,2pi]: "))
    interpolate(n_func, n, x)


if __name__ == '__main__':
    main()
