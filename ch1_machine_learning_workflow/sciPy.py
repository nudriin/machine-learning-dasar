from scipy import optimize


def f(x):
    return x**2 + 5 * x + 6


result = optimize.minimize(f, 0)
print(result)
