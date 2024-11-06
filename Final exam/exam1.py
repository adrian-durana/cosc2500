import numpy as np
import matplotlib.pyplot as plt

# polynomial and derivative functions
value = 2
def polynomial(x):
    return x**2 + 6*x
def derivative_one(x):
    return 2*x + 6

# step size
h_points = 1000
h = np.logspace(-16, -1, h_points) # array of x-values

# forward difference 
forward_errors = np.zeros(h_points) # array of y-values
for i in range(h_points):
    forward_difference = (polynomial(value + h[i]) - polynomial(value)) / h[i] # f-diff approx.
    forward_error = abs(forward_difference - derivative_one(value)) # forward error
    forward_errors[i] = forward_error # append onto forward error array

# central difference 
central_errors = np.zeros(h_points) # array of y-values
for i in range(h_points):
    central_difference = (polynomial(value + h[i]) - polynomial(value - h[i])) / (2*h[i]) # c-diff approx.
    central_error = abs(central_difference - derivative_one(value)) # central error
    central_errors[i] = central_error # append onto central error array

# plots and plot settings
plt.loglog(h, forward_errors, label="forward")
plt.loglog(h, central_errors, label="central")
plt.title("Error versus step size, x = " + str(value))
plt.ylabel("error")
plt.xlabel("step size (h)")
plt.legend()
plt.show()
