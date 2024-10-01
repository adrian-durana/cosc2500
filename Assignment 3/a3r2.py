import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#####################
## Shooting method ##
#####################

## ODE solver (finds initial value of first derivative)
def shooting(y_a, a=0, b=1, y0=0, y_b=np.exp(1)/3): # Initial guess of y'(a) = s
    def system(t,y): # system of first-order ODEs
        return [y[1], y[0] + (2/3)*np.exp(t)]
        # y[0]' = y[1], y[1]' = ...
    solution = sp.integrate.solve_ivp(system, [a,b], [y0,y_a], t_eval=[b])
    return float(solution.y[0, -1] - y_b) # difference between guess and y_b

## Guess two values s.t. product < 0
print(shooting(1))
print(shooting(0))

## Bisection method (Sauer, modified)
def bisection(a,b,iterationCap=50): # interval (a,b)
    if np.sign(shooting(a)) == np.sign(shooting(b)):
        raise ValueError("The condition f(a)f(b) < 0 is not satisfied.")
    iterations = 0
    while (b-a)/2 >= 10**(-9):
        iterations += 1
        c = (a+b)/2
        if iterations >= iterationCap: break
        elif shooting(a)*shooting(c) < 0: b = c
        else: a = c
    #print("The approximate value for y'(a) is " + str((a+b)/2) + '.')
    #print('The number of iterations required is ' + str(iterations) + '.')
    return int((a+b)/2)

## Store value of bisection method answer
b_ans = bisection(1, 0)

## Solve ODE numerically, plot
def system(t, y):
    y1, y2 = y  # y = y1, dy/dt = y2
    return [y2, y1 + 2/3 * np.exp(t)]  # dy1/dt = y2, dy2/dt = ...
y0 = [0, b_ans] # Initial conditions y(0), y'(0)
sol = sp.integrate.solve_ivp(system, [0, 1], [0, 1/3], rtol = 1e-9, atol = 1e-9, max_step = 1e-3)
t = sol.t
y = sol.y[0]
plt.plot(t,y, label = 'approx.')

## Exact solution of ODE, plot
def solution(x):
    return (1/3)*x*np.exp(x) 
x_values = np.linspace(0,1,num=100)
y_values = solution(x_values)
plt.plot(x_values, y_values, label = 'exact')


##############################
## Finite difference method ##
##############################

"""
Consider the following partition of [a,b] with spacing h:
    t[a] = t[0] < t[1] < ... < t[n+1] = t[b]
Let w_i = w(t_i) be an approximation for the correct values y_i = y(t_i) at discrete points t_i.
    
Substitute values into the numerical approximation for the second derivative, and h = 1/n+1:
    ( ( w[i+1] - 2*w[i] + w[i-1] ) / h**2 ) = w[i] + (2/3)*np.exp(t[i])
    w[i-1] - (2 + h**2)*w[i] + w[i+1] = (2/3)*(h**2)*np.exp(t[i])
    w[i-1] - (2 + (1/(n+1))**2)*w[i] + w[i+1] = (2/3)*((1/(n+1))**2)*np.exp(t[i])

Assume n = 3:
    w[0] + w[1] * (2 + (1/(n+1))**2) + w[2] = (2/3)*((1/(n+1))**2)*np.exp(t[1])
    w[1] + w[2] * (2 + (1/(n+1))**2) + w[3] = (2/3)*((1/(n+1))**2)*np.exp(t[2])
    w[2] + w[3] * (2 + (1/(n+1))**2) + w[4] = (2/3)*((1/(n+1))**2)*np.exp(t[3])

Insert boundary conditions: y(0) = w[0] =  0, y(1) = w[4] = e/3
    0 + w[1] * (-33/16) + w[2] = (1/24)*np.exp(t[1])
    w[1] + w[2] * (-33/16) + w[3] = (1/24)*np.exp(t[2])
    w[2] + w[3] * (-33/16) + (np.exp(1)/3) = (1/24)*np.exp(t[3])

Put into matrix equation
| -33/16  1     0   |  |w[1]|       | (1/24)*np.exp(t[1]) - 0              |
|    1  -33/16  1   |  |w[2]|   =   | (1/24)*np.exp(t[2])                  |
|   0     1  -33/16 |  |w[3]|       | (1/24)*np.exp(t[3]) -  (np.exp(1)/3) |
"""

# Preparation
n = 1000
h = 1 / (n+1)
A = np.zeros((n,n)) # 0th column is w[0]
c = np.zeros((n,1))
t_n = np.linspace(0,1,num=n)

# Create the matrix
for i in range(1,n):
    A[i-1, i-1] = -(2 + (1/(n+1))**2)
    A[i-1, i] = 1
    A[i, i-1] = 1
    c[i-1] = (2/3)*((1/(n+1))**2)*np.exp(t[i-1])
A[n-1,n-1] = -(2 + (1/(n+1))**2)

# Value of boundary conditions y(0), y(1)
c[0] = c[0] - 0 # subtract left boundary condition
c[n-1] = c[n-1] - np.exp(1)/3 # sub. r.b.c

# Solve and plot!
solution = np.linalg.solve(A, c)
print(solution)
plt.plot(t_n, solution, label = 'finite')

## Finish the plot
plt.legend()
plt.show()

"""
Task: generalise for all second-order linear ODEs
"""