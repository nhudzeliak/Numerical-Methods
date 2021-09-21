from polynomial_interpolation import interpolate


def main():
    n_func = int(input("Choose function:\n1 - 1/1+25x^2\n2 - ln(x+2)\n3 - polynomial"))
    a = float(input("Enter a: "))
    if n_func == 2:
        while a <= -2:
            a = float(input("Enter a (valid for ln):"))
    b = float(input("Enter b: "))
    while b <= a:
        b = float(input("Enter b > a: "))
    n = int(input("Enter n: "))
    while n <= 0:
        n = int(input("Enter n: "))
    x = float(input("Enter xє[a,b]: "))
    while x < a or x > b:
        x = float(input("Enter xє[a,b]: "))
    m = int(input("Enter m (>= n+1) - number of points for visualization: "))
    while m < n + 1:
        m = int(input("Enter m (>= n+1) - number of points for visualization: "))
    interpolate(n_func, a, b, n, x, m)


if __name__ == "__main__":
    main()
