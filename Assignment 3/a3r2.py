import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Shooting method

## F(s) equal to differnece betweens pecified boudnary value,a dn boudnaryt value caclualted for given set of initial condtiions s


## Start with initial guess for slope s_a, go along with initial value u_a

## Solve IVP that results from this initial slope, compare with boundary value y_b

## Change guess of initial slope until boundary value matches



## Find F(s) = 0 (root-finding, Newton's method)
# difference between y_b and y(b), y(t) is solution of IVP with y(a) = y_a, y'(a) = 

# Find two values of s such that F(s_0) F(s_1) < 0

#v' = 4y
#y' = v

def function(s, a = 0, b = 1): # Start with initial guess of slope at x=a
    y_b = (1/3)*np.exp(1) # The boundary condition
    def ydot(t,y):
        return [y[1], y[0] + (2/3)*np.exp(t)] # y' = v, v' = ...
        # y_1' = y_2, y_2 = 4y_1
        # y[0] is y1, the original variable, y[1] is y2, the derivative of original
    sol = sp.integrate.solve_ivp(ydot, [a,b], [0,s], t_eval=[b]) # function ydot, domain [a,b], initial conditions y(0) = 0, y(1) - s
    z = float(sol.y[0, -1] - y_b)
    return z

print(function(-1))
print(function(0))

## Bisection method (Sauer, translated and modified)
def bisection(a,b,iterationCap=50): # input: interval (a,b)
    if np.sign(function(a)) == np.sign(function(b)):
        raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
    iterations = 0
    while (b-a)/2 >= 10**(-9):
        iterations += 1
        c = (a+b)/2
        if iterations >= iterationCap: break
        elif function(a)*function(c) < 0: b = c
        else: a = c
    print('The approximate root is ' + str((a+b)/2) + '.')
    print('The number of iterations required is ' + str(iterations) + '.')

bisection(0, 1)

## Solve actual thing

def system(t, y):
    y1, y2 = y  # y1 = y, y2 = dy/dt
    dydt = [y2, y1 + 2/3 * np.exp(t)]  # dy1/dt = y2, dy2/dt = -y1
    return dydt

# Initial conditions: y(0), y'(0)
y0 = [0, 1/3]

sol = sp.integrate.solve_ivp(system, [0, 1], [0, 1/3], rtol = 1e-9, atol = 1e-9, max_step = 1e-3)

t = sol.t
y = sol.y[0]

plt.plot(t,y)
plt.show()

## Answer
