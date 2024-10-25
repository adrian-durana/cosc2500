import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Two-dimensional pursuit curve solver
With arclength to check lengths
"""

## Parameters
k = 0.999
x0, y0 = 0, 0
tmax = np.pi*18
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory
def T_x(t): return np.cos(t) #np.cos(2*t)*np.cos(t)
def T_y(t): return np.sin(t) #np.cos(2*t)*np.sin(t)
def T(t): return np.array([T_x(t), T_y(t)])

## Target derivative
def dT_x(t): return -np.sin(t) # -(2*np.sin(2*t)*np.cos(t)+np.cos(2*t)*np.sin(t))
def dT_y(t): return np.cos(t) # -2*np.sin(2*t)*np.sin(t)+np.cos(2*t)*np.cos(t)
def dT(t): return np.array([dT_x(t), dT_y(t)])

## Pursuer trajectory - system of ODEs
def pursuit_curve(t, P):
    x, y = P
    diff = T(t) - np.array([x, y])  # Distance vector
    norm_diff = np.linalg.norm(diff)  # Norm of vector
    direction = k * np.sqrt(np.sum(dT(t)**2)) * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1]]

## Event function
def proximity_event(t, P):
    x, y = P
    distance = np.linalg.norm(T(t) - np.array([x, y])) # Norm of distance vector 
    return distance - 0.001  # Trigger when distance is less than 1e-6
proximity_event.terminal = True  # Terminate integration if event occurs

## ODE solution
solution = solve_ivp(pursuit_curve, [0, tmax], [x0, y0], t_eval=t_eval, 
                     events=proximity_event)

## Event checker
if len(solution.t_events) > 0 and len(solution.t_events[0]) > 0:
    t_intersect = solution.t_events[0][0]
    mask = solution.t <= t_intersect # Mask all points up to intersection
else:
    mask = np.ones_like(solution.t, dtype=bool) # Mask all points

## Curve outputs
x_pursuer = solution.y[0][mask]
y_pursuer = solution.y[1][mask]
x_target = T(solution.t[mask])[0]
y_target = T(solution.t[mask])[1]

## Arclength calculation
def compute_arclength(x, y):
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(diffs)
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer)
arclength_target = compute_arclength(x_target, y_target)
print(f"Arclength of pursuer: {arclength_pursuer:.6f}")
print(f"Arclength of target: {arclength_target:.6f}")

## Analytic solution (1)
x_span = np.linspace(0, 2, int(1/0.000095)-1)
def true(x, x0=x0, y0=y0):
    eta = (x / x0)**2
    r0 = np.sqrt(x0**2 + y0**2)
    y = (1/4)*((y0+r0)*eta + (y0-r0)*np.log(eta) + 3*y0 - r0)
    return y
def false(x, x0=x0):
    y = (1/2)*(((x**2 - x0**2)/(2*x0)) - x0*np.log(x/x0))
    y = 0.5 * ((x**(1 + 1/k)) / (x0**(1/k) * (1 + 1/k)) - (x0**(1/k) * x**(1 - 1/k)) / (1 - 1/k)) + (x0 * k) / (k**2 - 1)
    return y
#plt.plot(x_span, true(x_span), color='green', label='Analytic solution') ## Analytic solution
#plt.plot(x_span, false(x_span), color='brown', label='Analytic solution')

## Plotting
plt.figure(figsize=(5, 5))
plt.plot(x_pursuer, y_pursuer, label='Pursuer curve', color='blue')
plt.scatter(x_pursuer[0], y_pursuer[0], color='blue')
plt.plot(x_target, y_target, label='Target curve', color='red')
plt.scatter(x_target[0], y_target[0], color='red')

if len(solution.t_events[0]) > 0:
    plt.scatter(x_pursuer[-1], y_pursuer[-1], marker='x', color='green', label='Intersection')
else:
    plt.scatter(x_pursuer[-1], y_pursuer[-1], marker='x', color='blue', label='Pursuer')
    plt.scatter(x_target[-1], y_target[-1], marker='x', color='red', label='Target')

plt.title('Pursuit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()