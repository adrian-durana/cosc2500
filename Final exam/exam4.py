import numpy as np
from scipy import integrate as sp
import matplotlib.pyplot as plt

## Intervals, boundary conditions, ODE
a, b = 0, 1
y_a, y_b = 0, 0
def system(t,y): return [y[1], y[0]**2 - t]

## Shooting method
def guess(s):
    solution = sp.solve_ivp(system, [a,b], [y_a, s], t_eval=[b], max_step = 1e-1)
    return float(solution.y[0,-1] - y_b)
def bisection(A, B, tol = 1e-9, maxiterations = 1000): # Root-finding
    if np.sign(guess(A)) == np.sign(guess(B)):
        raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
    for i in range(0, int(maxiterations)):
        c = (A+B)/2
        if guess(A)*guess(c) < 0: B = c
        else: A = c
        if np.abs(A-B) < tol: break
    return c
def shooting(g1, g2): # Solving the IVP
    sol = sp.solve_ivp(system, [a, b], [y_a, bisection(g1,g2)], max_step = 1e-3)
    plt.plot(sol.t, sol.y[0])
shooting(-10, 10)
plt.ylabel("y")
plt.xlabel("t")
plt.show() ## Plot