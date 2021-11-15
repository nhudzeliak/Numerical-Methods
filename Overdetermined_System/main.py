from overdetermined_system import solve_system


def main():
    x, err, norm = solve_system("input.txt")
    print(f"x = {x}\ne = {err}\ne~ = {norm}")


if __name__ == '__main__':
    main()