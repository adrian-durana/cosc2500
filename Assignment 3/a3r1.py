import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

## Functions
F = 1
def function(t,y):
    if F == 1: return t
    if F == 2: return 2*(t+1)*y
    if F == 3: return 1 / (y*y)

## Intervals, step size
interval_start = 0
interval_end = 1
y0 = 1

## Exact solutions
def exact(k, plot = 0, i1 = interval_start, i2 = interval_end):
    h = 0.1 * (2)**(-k)
    x_values = np.linspace(i1, i2, num = int((i2 - i1)/h))
    if F == 1: exact_values = 0.5 * x_values**2 + 1
    if F == 2: exact_values = 0.5 * np.exp(x_values**2 + 2*x_values + np.log(2))
    if F == 3: exact_values = (3*x_values + 1)**(1/3)
    if plot == 1: plt.plot(x_values, exact_values, label = "Exact solution")
    return exact_values[-1]

## Euler's method
def euler(k, plot = 0, y0 = y0, i1 = interval_start, i2 = interval_end):
    h = 0.1 * (2)**(-k)
    t, y, n = [i1], [y0], int((i2-i1)/h)
    for i in range(0, n-1):
        t.append(t[i]+h)
        y.append(y[i]+h*function(t[i],y[i]))
    if plot == 1: plt.plot(t,y, label = "Euler")
    return y[-1]

## Fixed step Runge-Kutta method
def rk_f(k, plot = 0, y0 = y0, i1 = interval_start, i2 = interval_end):
    h = 0.1 * (2)**(-k)
    n = int((i2-i1)/h)
    t = np.transpose(np.linspace(i1, i2, n))
    y = [y0]
    for n in range(1, n):
        k1 = h*function(t[n-1], y[n-1])
        k2 = h*function(t[n-1] + 0.5*h, y[n-1] + 0.5*k1)
        k3 = h*function(t[n-1] + 0.5*h, y[n-1] + 0.5*k2)
        k4 = h*function(t[n-1] + h, y[n-1] + k3)
        y.append(y[n-1] + (1/6)*(k1 + 2*k2 + 2*k3 + k4))
    y = np.transpose(y)
    if plot == 1: plt.plot(t,y, label = "Fixed R-K")
    return y[-1]

## Log-log plot of error
def error():
    A, B = np.zeros(6), np.zeros(6)
    eval, H = np.zeros(6), np.zeros(6)
    for k in range(6):
        H[k] = (0.1 * (2)**(-k))
        eval[k] = exact(k)
        A[k] = euler(k)
        B[k] = rk_f(k)
    eu = np.subtract(eval, A)
    rk = np.subtract(eval, B)
    plt.loglog(H, eu, label = "Euler")
    plt.loglog(H, rk, label = "Runge-Kutta")

## Adaptive step Runge-Kutta method
def rk_a(y0 = y0, i1 = interval_start, i2 = interval_end):
    sol = sp.integrate.solve_ivp(function, [interval_start, interval_end], [y0], rtol = 1e-9, atol = 1e-9, max_step = 1e-3)
    sol_x = sol.t
    sol_y = sol.y[0]
    plt.plot(sol_x, sol_y, label = "Adaptive R-K")

## Plots
exact(0, 1)
euler(0, 1)
rk_f(0, 1)
rk_a()

error()

plt.legend() 
plt.show()



