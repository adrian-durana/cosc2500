## Function 1: f(x) = cos(x), roots at (pi/2), (3pi/2)
    
## Function 2: f(x) = (x+2)(x-1)^3, roots at (-2), (1)

## Compare by (a) the number of iterations required to find the roots to some target accuracy
            # (b) the difficulty of choosing starting points or intervals that give convergence

import numpy as np
import matplotlib.pyplot as plt

## Function and root
root = 1*np.pi/2
def function(x):
    return np.cos(x)

## Bisection method (Sauer, translated)
def bisection(a,b): # input: starting interval (a,b)
    if np.sign(function(a)) == np.sign(function(b)):
        raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
    iterations = 0
    while (b-a)/2 >= 10**(-9):
        iterations += 1
        c = (a+b)/2
        if function(c) == 0: break
        elif function(a)*function(c) < 0: b = c
        else: a = c
    return iterations
    ## print('The approximate root is ' + str((a+b)/2) + '.')
    ## print('The number of iterations required is ' + str(iterations) + '.')
    
## Plot: bisection (fixed midpoint, changing radius)
def bisection_plot1(points):
    x_values = np.logspace(-9, np.log10(np.pi), num=points) # For cos(x)
    y_values = []
    interval_left = np.array([root]*points) - x_values
    interval_right = np.array([root]*points) + x_values
    for i in range(0, points):
        n = bisection(interval_left[i], interval_right[i])
        y_values.append(n)
    plt.semilogx(x_values, y_values)
    # plt.title("Interval radius versus number of iterations required, fixed midpoint, " + str(points) + "points")
    plt.ylabel("Number of iterations")
    plt.xlabel("Interval radius")
    plt.show() 

## Plot: bisection (changing midpoint, fixed radius)
def bisection_plot2(points, radius):
    x_values = np.linspace((root-(0.999*radius)),(root+(0.999*radius)), num=points)
    y_values = []
    interval_left = x_values - np.array([radius]*points)
    interval_right = x_values + np.array([radius]*points)
    for i in range(0, points):
        n = bisection(interval_left[i], interval_right[i])
        y_values.append(n)
    plt.plot(x_values, y_values)
    # plt.title("Midpoint location versus number of iterations required, fixed radius, " + str(points) + "points")
    plt.ylabel("Number of iterations")
    plt.xlabel("Midpoint location")
    plt.show() 

## Fixed-point iteration (Sauer, translated and modified)
def iteration(x): # input: x (starting guess)
    iterations = 0
    while np.abs(x-root) >= 10**(-9):
        iterations += 1
        if iterations >= 100: break
        x = function(x) + x
    print('The approximate root is ' + str(x) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')

## Newton's method, numerical derivative
def newton(x): # input: x (starting guess)
    h = 6.06*10**(-6)
    iterations = 0
    while np.abs(x-root) >= 10**(-9):
        iterations += 1
        if iterations >= 100: break
        x = x - (function(x) / ((function(x+h) - function(x-h)) / (2*h)))
    print('The approximate root is ' + str(x) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')

# bisection(1.561, 1.581)
# iteration(3*np.pi/2- 10**(-12))
# newton(2.73635)
bisection_plot1(10000)
