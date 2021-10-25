from spline_interpolation import spline_interpolation


def main():
    a = float(input("Enter a: "))
    b = float(input("Enter b: "))
    n = int(input("Enter n: "))
    q = int(input("Enter N (for plotting): "))
    spline_interpolation(a, b, n, q)


if __name__ == "__main__":
    main()
