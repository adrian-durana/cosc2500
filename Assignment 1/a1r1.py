import numpy as np
import matplotlib.pyplot as plt

# machine epsilon
machine_eps = np.finfo(float).eps # 2.220446049250313e-16
machine_eps_exponent = np.floor(np.log10(machine_eps))
rot_2 = (np.finfo(float).eps)**(1/2) # 1.4901161193847656e-08
rot_3 = (machine_eps)**(1/3) # 6.055454452393343e-06

# polynomial and derivative functions
value = 1.5
def polynomial(x):
    return x**4 - 2*x**3
def derivative_one(x):
    return 4*x**3 - 6*x**2
def derivative_two(x):
    return 12*x**2 - 12*x
def derivative_three(x):
    return 24*x

# step size
h_points = 1000
h = np.logspace(-16, -1, h_points) # array of x-values

# randomness
random = np.zeros(h_points)
#random = 0.001*np.random.randn(h_points)

# forward difference 
forward_errors = np.zeros(h_points) # array of y-values
for i in range(h_points):
    forward_difference = (polynomial(value + random[i] + h[i]) - polynomial(value + random[i])) / h[i] # f-diff approx.
    forward_error = abs(forward_difference - derivative_one(value)) # forward error
    forward_errors[i] = forward_error # append onto forward error array

# backward difference 
backward_errors = np.zeros(h_points) # array of y-values
for i in range(h_points):
    backward_difference = (polynomial(value + random[i]) - polynomial(value + random[i] - h[i])) / h[i] # b-diff approx.
    backward_error = abs(backward_difference - derivative_one(value)) # backward error
    backward_errors[i] = backward_error # append onto backward error array

# true error (forward, backward)
true_errors = np.zeros(h_points)
for i in range(h_points):
    true_error = abs((h[i]/2)*derivative_two(value))
    true_errors[i] = true_error

# central difference 
central_errors = np.zeros(h_points) # array of y-values
for i in range(h_points):
    central_difference = (polynomial(value + random[i] +h[i]) - polynomial(value + random[i] - h[i])) / (2*h[i]) # c-diff approx.
    central_error = abs(central_difference - derivative_one(value)) # central error
    central_errors[i] = central_error # append onto central error array

# true error (central)
truec_errors = np.zeros(h_points)
for i in range(h_points):
    truec_error = abs( ( ( (h[i])**2) /6) * derivative_three(value) )
    truec_errors[i] = truec_error

# plots and plot settings
plt.loglog(h, forward_errors, label="forward")
plt.loglog(h, backward_errors, label="backward")
plt.loglog(h, central_errors, label="central")
plt.title("Error versus step size, x = " + str(value) + ", with added error")
plt.ylabel("error")
plt.xlabel("step size (h)")
plt.legend()
plt.show()

# plt.loglog(h, true_errors)
# plt.loglog(h, truec_errors)
# plt.loglog(sauer, abs(((polynomial(value + sauer) - polynomial(value - sauer)) / (2*sauer)) - derivative_one(value)), 'bo')