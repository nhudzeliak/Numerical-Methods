from best_approximation import best_approximation


def main():
    n_func = int(input("Choose function:\n1 - x ^ 2\n2 - 3x^3 - 1 "))
    n = int(input("Enter n: "))
    a = float(input("Enter a: "))
    b = float(input("Enter b: "))
    best_approximation(n, a, b, n_func)


if __name__ == "__main__":
    main()
