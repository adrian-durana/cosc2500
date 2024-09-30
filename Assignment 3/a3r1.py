import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

## Functions
def function(t,y):
    # return t
    #return 2*(t+1)*y
    return 1/(y**2)

## Intervals, step size
interval_start = 0
interval_end = 1
y0 = 1

## Exact solutions
def exact(k, i1 = interval_start, i2 = interval_end):
    h = 0.1 * (2)**(-k)
    x_values = np.linspace(i1, i2, num = int((i2 - i1)/h))
    #exact_values = 0.5 * x_values**2 + 1
    #exact_values = 0.5 * np.exp(x_values**2 + 2*x_values + np.log(2))
    exact_values = (3*x_values + 1)**(1/3)
    ## plt.plot(x_values, exact_values, label = "Exact solution")
    return exact_values[-1]

## Euler's method
def euler(k, y0 = y0, i1 = interval_start, i2 = interval_end):
    h = 0.1 * (2)**(-k)
    t, y, n = [i1], [y0], int((i2-i1)/h)
    for i in range(0, n-1):
        t.append(t[i]+h)
        y.append(y[i]+h*function(t[i],y[i]))
    ## plt.plot(t,y, label = "Euler")
    return y[-1]

## Fixed step Runge-Kutta method
def rk_f(k, y0 = y0, i1 = interval_start, i2 = interval_end):
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
    ## plt.plot(t,y, label = "Fixed R-K")
    return y[-1]

## Log-log plot of error
a, b, c, d = [], [], [], []
for k in range(0,6):
    d.append(0.1 * (2)**(-k))
    a.append(exact(k))
    b.append(euler(k))
    c.append(rk_f(k))
e = np.subtract(a, b)
f = np.subtract(a, c)
plt.loglog(d, e, label = "Euler")
plt.loglog(d, f, label = "Runge-Kutta")


## Adaptive step Runge-Kutta method
def rk_a(y0 = y0, i1 = interval_start, i2 = interval_end):
    sol = sp.integrate.solve_ivp(function, [interval_start, interval_end], [y0], rtol = 1e-9, atol = 1e-9, max_step = 1e-3)
    sol_x = sol.t
    sol_y = sol.y[0]
    plt.plot(sol_x, sol_y, label = "Adaptive R-K")


## Plots

#exact()
#euler()
#rk_f()
#rk_a()
plt.legend() 
plt.show()



