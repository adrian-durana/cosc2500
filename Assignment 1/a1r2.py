import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# number of points - start and end values
points_start = 10**1
points_end = 10**5
points_interval = 10

# x and y arrays
step_size = []
trapezoid = []
simpson = []

# calculating integrals
for i in range(points_start, points_end, points_interval):
    x1 = np.linspace(0, np.pi/2, num=i)
    #x1 = np.linspace(0, 10, num=i)
    y1 = np.cos(x1) + 0.001*np.random.randn()
    #y1 = np.sin(x1**2) + 1 + 0.001*np.random.randn()
    f_trap = abs(1 - sp.trapezoid(y1, x=x1))
    f_simp = abs(1 - sp.simpson(y1, x=x1))
    step_size.append(np.pi/2 / i)
    trapezoid.append(f_trap)
    simpson.append(f_simp)

# plot
plt.loglog(step_size, trapezoid, label="Trapezoid rule")
plt.loglog(step_size, simpson, label="Simpson's rule")
plt.title("Trapezoid vs. Simpson's rule, f(x) = cos(x), x to pi/2, added error")
plt.ylabel("error")
plt.xlabel("step size (h)")
plt.legend()
plt.show()

# adaptive step
def int1(x):
    return np.cos(x)
def int2(x):
    return np.sin(x**2) + 1
result1, error1 = sp.quad(int1, 0, np.pi)
result2, error2 = sp.quad(int2, 0, 10)
print("Function 1:" + str(result1)+ "," + str(error1))
print("Function 2:" + str(result2)+ "," + str(error2))