## Function 1: f(x) = cos(x), roots at (pi/2), (3pi/2)
    
## Function 2: f(x) = (x+2)(x-1)^3, roots at (-2), (1)

## Compare by (a) the number of iterations required to find the roots to some target accuracy
            # (b) the difficulty of choosing starting points or intervals that give convergence

import numpy as np
import matplotlib.pyplot as plt

## Function and root, iteration cap
root = 1
def function(x):
    #return np.cos(x)
    return (x+2)*(x-1)*(x-1)*(x-1)
iterationCap = 50

## Bisection method (Sauer, translated and modified)
def bisection(a,b): # input: interval (a,b)
    if np.sign(function(a)) == np.sign(function(b)):
        raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
    iterations = 0
    while (b-a)/2 >= 10**(-9):
        iterations += 1
        if iterations >= iterationCap: break
        c = (a+b)/2
        if np.abs(c-root) <= 10**(-9): break
        elif function(a)*function(c) < 0: b = c
        else: a = c
    print('The approximate root is ' + str((a+b)/2) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')
    return iterations

## Plot: bisection
def bisection_plot(points, radius): # number of points, length of interval radius
    x_values = np.linspace((root-((1-10**(-6))*radius)),(root+((1-10**(-6))*radius)), num=points) 
    y_values = []
    interval_left = x_values - np.array([radius]*points)
    interval_right = x_values + np.array([radius]*points)
    for i in range(0, points):
        n = bisection(interval_left[i], interval_right[i])
        y_values.append(n)
    plt.plot(x_values, y_values)
    plt.title("Figure a: Root = " + str(root) + ", Interval radius " + str(round(radius,3)))
    plt.ylabel("Number of iterations")
    plt.xlabel("Midpoint location")

## Fixed-point iteration (Sauer, translated and modified)
def iteration(x): # input: x (starting guess)
    iterations = 0
    while np.abs(x-root) >= 10**(-9):
        iterations += 1
        if iterations >= 100: break
        x = function(x) + x
    print('The approximate root is ' + str(x) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')
    return iterations

## Plot: fixed-point iteration
def iteration_plot(points):
    x_values = np.linspace((root - 0.5 + 10**(-6)), (root + 0.5 - 10**(-6)), points)
    y_values = []
    for i in range(0, points):
        n = iteration(x_values[i])
        y_values.append(n)
    plt.plot(x_values, y_values)
    plt.ylabel("Number of iterations")
    plt.xlabel("Initial value")
    plt.show() 

## Newton's method, numerical derivative
def newton(x): # input: x (starting guess)
    h = 6.06*10**(-6)
    iterations = 0
    while np.abs(x-root) >= 10**(-6):
        iterations += 1
        if iterations >= 100: break
        x = x - (function(x) / ((function(x+h) - function(x-h)) / (2*h)))
    print('The approximate root is ' + str(x) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')
    return iterations

## Plot: Newton's method
def newton_plot(points):
    x_values = np.linspace((root - 0.5), (root + 0.5), points)
    y_values = []
    for i in range(0, points):
        n = newton(x_values[i])
        y_values.append(n)
    plt.plot(x_values, y_values)
    plt.ylabel("Number of iterations")
    plt.xlabel("Initial value")
    plt.show()