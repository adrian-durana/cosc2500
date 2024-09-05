## Compare by (a) the number of iterations required to find the roots to some target accuracy
            # (b) the difficulty of choosing starting points or intervals that give convergence

import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

## Function (used derivative calculator)
def function(x):
    return np.exp(-1*x**2)*(x+(3/13))*(x-(7/11))**2

## Golden section search (unimodal f, minimum in a,b, number of steps k) (Sauer)
def golden(a,b,k):
    g = (np.sqrt(5)-1)/2
    x1 = a+(1-g)*(b-a)
    x2 = a+g*(b-a)
    f1 = function(x1)
    f2 = function(x2)
    for i in range(1, k):
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a+(1-g)*(b-a)
            f2 = f1
            f1 = function(x1)
        else:
            a = x1
            x1 = x2
            x2 = a+g*(b-a)
            f1 = f2
            f2 = function(x2)
    y = (a+b)/2
    return y

## Successive parabolic interpretation (initial guesses r,s,t, number of steps k) (Sauer)
def parabola(r,s,t,k):
    x = [r,s,t]
    x[0] = r
    x[1] = s
    x[2] = t
    fr = function(r)
    fs = function(s)
    ft = function(t)
    for i in range(3, k+2):
        new = (r+s)/2-(fs-fr)*(t-r)*(t-s)/(2*((s-r)*(ft-fs)-(fs-fr)*(t-s)))
        x.append(new)
        t = s
        s = r
        r = x[i]
        ft = fs
        fs = fr
        fr = function(r)
    return x[-1]

## Newton's method
def derivative_one(x):
    return -((3146*x**4 - 3278*x**3 - 4369*x**2 + 3572*x - 175) * np.exp(-1*x**2)) / 1573
def derivative_two(x):
    return ((6292*x**5 - 6556*x**4 - 21322*x**3 + 16978*x**2 + 8388*x - 3572) * np.exp(-1*x**2)) / 1573
def newton(x, k):
    array = [x]
    for i in range(1,k):
        array.append(array[i-1] - (derivative_one(array[i-1]))/(derivative_two(array[i-1])))
    ans = array[k-1]
    return ans

## Secant method
def secant(x, k):
    array = [x]
    h = 6.06*10**(-6)
    for i in range(1,k):
        numder_one = (function(array[i-1]+h) - function(array[i-1]-h)) / (2*h)
        numder_two = (function(array[i-1]+h)+function(array[i-1]-h)-2*function(array[i-1])) / (h**2)
        array.append(array[i-1] - (numder_one/numder_two))
    ans = array[k-1]
    return ans