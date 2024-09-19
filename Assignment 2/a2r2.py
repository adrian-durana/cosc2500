## Compare by (a) the number of iterations required to find the roots to some target accuracy
            # (b) the difficulty of choosing starting points or intervals that give convergence

import numpy as np
import matplotlib.pyplot as plt

## Function (used derivative calculator)
def function(x):
    return np.exp(-1*x**2)*(x+(3/13))*(x-(7/11))**2

## Graph
x_values = np.linspace(-5, 5, num=1000)
plt.plot(x_values, function(x_values))
plt.plot(x_values, np.zeros(1000), color='black')
plt.xlim(-5, 5)
plt.ylim(-1, 1)
#plt.show()

## Golden section search (unimodal f, minimum in a,b, number of steps k) (Sauer, modified)
def golden(a,b, tol=1e-9, maxiterations = 1000):
    g = (np.sqrt(5)-1)/2
    x1, x2 = a+(1-g)*(b-a), a+g*(b-a)
    f1, f2 = function(x1), function(x2)
    for i in range(0, maxiterations):
        if np.abs(a-b) <= tol: break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a+(1-g)*(b-a)
            f1 = function(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a+g*(b-a)
            f2 = function(x2)
    y = (a+b)/2
    print("The approximate minima is " + str(y) + '.')
    print("The number of iterations required is " + str(i+1) + '.')

## Successive parabolic interpolation (initial guesses r,s,t, number of steps k) (Sauer)
def parabola(r,s,t, tol=1e-9, maxiterations = 1000):
    fr, fs, ft = function(r), function(s), function(t)
    for i in range(0, maxiterations):
        if np.abs(r-s) <= tol: break
        n = (r+s)/2-(fs-fr)*(t-r)*(t-s)/(2*((s-r)*(ft-fs)-(fs-fr)*(t-s)))
        t, s, r = s, r, n
        ft, fs = fs, fr
        fr = function(r)
    print("The approximate minima is " + str(r) + '.')
    print("The number of iterations required is " + str(i+1) + '.')

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