## Function 1: f(x) = cos(x), roots at (pi/2), (3pi/2)
    
## Function 2: f(x) = (x+2)(x-1)^3, roots at (-2), (1)

## Compare by (a) the number of iterations required to find the roots to some target accuracy
            # (b) the difficulty of choosing starting points or intervals that give convergence

import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

## Function
def function(x):
    return np.cos(x)
def derivative(x):
    return -np.sin(x)

## Bisection method (Sauer)
def bisection(a,b): # input: starting interval (a,b)
    if np.sign(function(a)*function(b)) > 0:
        raise Exception("The condition f(a)f(b) < 0 is not satisfied.")
    while (b-a)/2 >= 10**(-6):
        c = (a+b)/2
        if function(c) == 0: break
        elif function(a)*function(c) < 0: b = c
        else: a = c
    print('The final interval ['+str(a)+','+str(b)+'] contains a root.')
    print('The approximate root is' + str((a+b)/2) + '.')

## Fixed-point iteration (Sauer)
def iteration(x, k): # input: x (starting guess), k (iteration steps)
    array = [x]
    for i in range(1,k):
        array.append(function(array[i-1]) + array[i-1])
    ans = array[k-1]
    print(ans)

## Newton's method
def newton(x, k):
    array = [x]
    for i in range(1,k):
        array.append(array[i-1] - (function(array[i-1]))/(derivative(array[i-1])))
    ans = array[k-1]
    print(ans)

## Secant method
def secant(x, k):
    array = [x]
    h = 6.06*10**(-6)
    for i in range(1,k):
        numder = (function(array[i-1]+h) - function(array[i-1]-h)) / (2*h)
        array.append(array[i-1] - (function(array[i-1]))/numder)
    ans = array[k-1]
    print(ans)