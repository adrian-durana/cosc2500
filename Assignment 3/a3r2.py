import numpy as np
from scipy import integrate as sp
import matplotlib.pyplot as plt

## Variables
a, b = 0, 1

F = 3
if F == 1: y_a, y_b = 0, np.exp(1)/3
elif F == 2: y_a, y_b = 1, np.exp(1)
elif F == 3: y_a, y_b = np.exp(1)**3, 1
else: raise ValueError("Check F")

#####################
## Shooting method ##
#####################

## System of first-order ODEs
def system(t,y): # y[0]' = y[1], y[1]' = ...
    if F == 1: return [y[1], y[0] + 2*np.exp(t)/3] 
    elif F == 2: return [y[1], y[0]*(2 + (4*t**2))]
    elif F == 3: return [y[1], 3*y[0] - 2*y[1]]

## Guessing second initial condition
def sGuess(s):
    solution = sp.solve_ivp(system, [a,b], [y_a, s], t_eval=[b])
    return float(solution.y[0,-1] - y_b)

## Solving ODE using bisection
def sSolve(g1, g2):
    def bisection(A, B, tol = 1e-9, maxiterations = 1000):
        if np.sign(sGuess(A)) == np.sign(sGuess(B)):
            raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
        for i in range(0, int(maxiterations)):
            c = (A+B)/2
            if sGuess(A)*sGuess(c) < 0: B = c
            else: A = c
            if np.abs(A-B) < tol: break
        return c
    sol = sp.solve_ivp(system, [a, b], [y_a, bisection(g1,g2)], rtol = 1e-9, atol = 1e-9, max_step = 1e-3)
    plt.plot(sol.t, sol.y[0], label = 'Shooting method')

##############################
## Finite difference method ##
##############################

def finite(n):
    A = np.zeros((n,n))
    c = np.zeros(n)
    t = np.linspace(a,b,num=n)

    if F == 1: factor = (1/(n+1))**2
    elif F == 2: factor = (2+4*t[i-1]**2) / (n+1)**2
    elif F == 3: factor = 3 / (n+1)**2

    for i in range(1,n):
        A[i-1, i] = A[i, i-1] = 1
        if F == 1:
            A[i-1, i-1] = -2 - factor
            c[i-1] = (2/3)*factor*np.exp(t[i-1])
        elif F == 2:
            A[i-1, i-1] = -2 - factor
        elif F == 3:
            A[i-1, i-1] = -2 - factor
            A[i-1, i] = (1 + 1 / (n+1))
            A[i, i-1] = (1 - 1 / (n+1))
    A[n-1,n-1] = -2 - factor
    
    if F == 3: 
        c[0] -= (1 - 1 / (n+1))*y_a
        c[n-1] -= (1 + 1 / (n+1))*y_b
    else: 
        c[0] -= y_a
        c[n-1] -= y_b

    solution = np.linalg.solve(A, c)
    plt.plot(t, solution, label = 'Finite difference')

## Plotting
plt.legend()
plt.show()

# Call functions here
sSolve(-100, 100)
finite(1000)

