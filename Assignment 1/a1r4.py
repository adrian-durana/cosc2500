import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# upper bound values
bound_start = 25
bound_end = 5*10**3
bound_interval = 1

# step size vs. number of steps
step_size = 0.01  # fixed step size
steps_num = 10**3    # fixed number of steps

# x and y arrays
bound = []
trapezoid1 = []
trapezoid2 = []

# calculating integral (fixed step size)
for i in range(bound_start, bound_end, bound_interval):
    a = int(np.ceil(i/step_size))
    x1 = np.linspace(0, i, num=a)
    x2 = np.linspace(0, i ,num=steps_num)
    y1 = np.exp(-x1)
    y2 = np.exp(-x2)
    f_trap1 = abs(1 - sp.trapezoid(y1, x=x1))
    f_trap2 = abs(1 - sp.trapezoid(y2, x=x2))
    bound.append(i)
    trapezoid1.append(f_trap1)
    trapezoid2.append(f_trap2)

# plot
plt.plot(bound, trapezoid1, label="Fixed step size")
#plt.plot(bound, trapezoid2, label="Fixed number of steps")
plt.title("Trapezoid rule over infinite interval, f(x) = exp(-x)")
plt.ylabel("error")
plt.xlabel("upper bound")
plt.legend()
plt.show()