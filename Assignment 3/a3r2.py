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

## Finish the plot
plt.legend()
plt.show()


##############################
## Finite difference method ##
##############################

## Numerical second derivative
def second_derivative():
    z =
    return z
# numder_two = (w[i+1] - 2*w[i] + w[i-1]) / (h**2) - w[i] + (2/3) * np.exp(t) = 0

"""
Consider the following partition of [a,b] with spacing h:
    t[a] = t[0] < t[1] < ... < t[n+1] = t[b]

Let w_i = w(t_i) be an approximation for the correct values y_i = y(t_i) at discrete points t_i.

Substitute values into the numerical approximation for the second derivative:
    

"""



# = (w[i+1] - 2*w[i] + w[i-1])  - (h**2)w[i] + (h**2)*(2/3) * np.exp(t_i) = 0

# = w[i-1] + w[i] (-2 - (h**2)) + w[i+1] + (h**2)*(2/3) * np.exp(t_i) = 0

# For n = 3, interval size is 1/4 (1/n+1), three equations

# Insert boundary conditions:

# y(0) = w_0 =  0, y(1) = w_4 =  e/3

# w[0] + w[1] (-2 - (h**2)) + w[2] + (h**2)*(2/3) * np.exp(t_1) = 0
# w[1] + w[2] (-2 - (h**2)) + w[3] + (h**2)*(2/3) * np.exp(t_2) = 0
# w[2] + w[3] (-2 - (h**2)) + w[4] + (h**2)*(2/3) * np.exp(t_3) = 0
    
# 0 + w[1] (-2 - (h**2)) + w[2] + (h**2)*(2/3) * np.exp(t_1) = 0
# w[1] + w[2] (-2 - (h**2)) + w[3] + (h**2)*(2/3) * np.exp(t_2) = 0
# w[2] + w[3] (-2 - (h**2)) + e/3 + (h**2)*(2/3) * np.exp(t_3) = 0

# Then substitute for h (1/4):

# 0 + w[1] (-33/16) + w[2] + (1/24) * np.exp(t_1) = 0
# w[1] + w[2] (-33/16) + w[3] + (1/24) * np.exp(t_2) = 0
# w[2] + w[3] (-33/16) + e/3 + (1/24) * np.exp(t_3) = 0

# Put into matrix equation

#| -33/16  1     0   |  |w[1]|       | - (1/24) * np.exp(t_1)        |
#|    1  -33/16  1   |  |w[2]|   =   | - (1/24) * np.exp(t_2)        |
#|   0     1  -33/16 |  |w[3]|       | - (1/24) * np.exp(t_3) - e/3  |


# Solve by gaussian elimination to find solution values at three points

# For smaller errors, use larger n

## TRANSLATION OF MATLAB CODE HINT!

# How many point do we want?
#n = 10

# Pre-allocate memory for matrix and vector
#A = np.zeros((n,n))
#c = np.zeros((n,1))

#tn = np.linspace(0,1,num=n) # Must be changed for 2a, since it goes to 1.5 ???

#h = t[2] - t[1] ### TRANSLATE FROM MATLAB TO PYTHON

# Boundary conditions
#A[0,0] = 1
#c[0] - 0 # First bc
#A[n-1,n-1] = 1
#c[n-1] = np.exp(1)/3 # Second bc

# Create the matrix

#for n in range(2, n):
#    A[n-1, n-2] = 1 / h**2
#    A[n-1, n-1] = 1/ h - 2/ h**2
#    A[n-1, n] = 2/ h ^2 - 1/ h
#    c[n-1] = tn(n) * np.sin(tn(n))

# Solve and plot!

