import numpy as np
import matplotlib.pyplot as plt

## Function (used derivative calculator)
def function(x): return np.exp(-(x**2))*5*x**3

## Graph
x_values = np.linspace(-5, 5, num=1000)
plt.plot(x_values, function(x_values))
plt.plot(x_values, np.zeros(1000), color='black')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.ylabel("y")
plt.xlabel("x")
plt.show()

## Newton's method, numerical derivative
def newton(x, tol=1e-9, maxiterations = 1000): # input: x (starting guess)
    h = 6.06*10**(-6)
    for i in range(maxiterations):
        numder_one = ((function(x+h) - function(x-h)) / (2*h))
        numder_two = (function(x+h) + function(x-h) - 2*function(x)) / (h**2)
        y = x - (numder_one / numder_two)
        if np.abs(x-y) <= tol: break
        x = y
    print('The approximate root is ' + str(x) + '.')
    print('The number of iterations required is ' + str(i+1) + '.')

newton(1.5)
