import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Two-dimensional pursuit curve solver
With arclength to check lengths
"""

## Parameters
k = 2
x0, y0 = 5, 0
tmax = 5
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory
def T_x(t):
    return 0*t
def T_y(t):
    return t
def T(t):
    return np.array([T_x(t), T_y(t)])

## Pursuer trajectory - system of ODEs
def pursuit_curve(t, P):
    x, y = P
    diff = T(t) - np.array([x, y]) # Distance vector
    norm_diff = np.linalg.norm(diff) # Norm of vector
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1]]

## ODE solution
solution = solve_ivp(pursuit_curve, [0, tmax], [x0, y0], t_eval=t_eval)
x_pursuer = solution.y[0]
y_pursuer = solution.y[1]
x_target = T(t_eval)[0]
y_target = T(t_eval)[1]

## Arclength calculation
def compute_arclength(x, y):
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(diffs)
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer)
arclength_target = compute_arclength(x_target, y_target)

## Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_pursuer, y_pursuer, label='Pursuer', color='blue')
plt.scatter(x_pursuer[0], y_pursuer[0], color='blue')
plt.plot(x_target, y_target, label='Target', color='red')
plt.scatter(x_target[0], y_target[0], color='red')
plt.title('Pursuit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid()

## Output the arclengths
print(f"Arclength of pursuer: {arclength_pursuer:.6f}")
print(f"Arclength of target: {arclength_target:.6f}")

plt.show()

## Real solution
#x_span = np.linspace(0.001, 5, int(1/0.001)-1)
#def true(x, x0=x0, y0=y0):
    #eta = (x / x0)**2
    #r0 = np.sqrt(x0**2 + y0**2)
    #chi = ((r0+y0)/(r0-y0))
    #y = (1/4)*((y0+r0)*eta + (y0-r0)*np.log(eta) + 3*y0 - r0)
    #return y
#plt.plot(x_span, true(x_span), color='green') ## Real solution