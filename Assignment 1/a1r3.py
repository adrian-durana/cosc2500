import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# function
def function(x):
    if 1 <= x <= 2:
        return np.exp(x-1) + np.exp(2-x)
    else: return 0

# number of points - start and end values
points_start = 10**1
points_end = 10**5
points_interval = 10**2

# x and y arrays
step_size = []
trapezoid = []

# calculating integral
for i in range(points_start, points_end, points_interval):
    x1 = np.linspace(0, 3, num=i)
    y1 = np.array([function(x) for x in x1])
    f_trap = abs(sp.trapezoid(y1, x=x1) - ((2 * np.e) - 2))
    step_size.append(3 / i)
    trapezoid.append(f_trap)

# plot
plt.loglog(step_size, trapezoid, label="Trapezoid rule")
plt.title("Trapezoidal integration over discontinuities")
plt.ylabel("error")
plt.xlabel("step size (h)")
plt.legend()
plt.show()