import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"""
Three-dimensional pursuit curve solver
With arclength to check lengths
"""

## Parameters
k = 2
x0, y0, z0 = 5, 0, 0
tmax = 5
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory
def T_x(t): return t
def T_y(t): return t**2
def T_z(t): return 0*t 
def T(t): return np.array([T_x(t), T_y(t), T_z(t)])

## Target derivative
def dT_x(t): return 0*t + 1
def dT_y(t): return 2*t
def dT_z(t): return 0*t
def dT(t): return np.array([dT_x(t), dT_y(t), dT_z(t)])

## Pursuer trajectory - system of ODEs 
def pursuit_curve(t, P):
    x, y, z = P
    diff = T(t) - np.array([x, y, z]) # Distance vector
    norm_diff = np.linalg.norm(diff) # Norm of vector
    direction = k * np.sqrt(np.sum(dT(t)**2)) * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1], direction[2]]

## Event function
def proximity_event(t, P):
    x, y, z = P
    distance = np.linalg.norm(T(t) - np.array([x, y, z])) # Norm of distance vector 
    return distance - 0.01  # Trigger when distance is less than 1e-6
proximity_event.terminal = True  # Terminate integration if event occurs

## ODE solution
solution = solve_ivp(pursuit_curve, [0, tmax], [x0, y0, z0], t_eval=t_eval,
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
z_pursuer = solution.y[2][mask]
x_target = T(solution.t[mask])[0]
y_target = T(solution.t[mask])[1]
z_target = T(solution.t[mask])[2]

## Arclength calculation
def compute_arclength(x, y, z):
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    return np.sum(diffs)
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer, z_pursuer)
arclength_target = compute_arclength(x_target, y_target, z_target)
print(f"Arclength of pursuer: {arclength_pursuer:.2f}")
print(f"Arclength of target: {arclength_target:.2f}")

## Plotting
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
# Pursuer #
ax.plot(x_pursuer, y_pursuer, z_pursuer, color='blue')
ax.scatter(x_pursuer[0], y_pursuer[0], z_pursuer[0], color='blue', label='Pursuer Start')
# Target #
ax.plot(x_target, y_target, z_target, color='red')
ax.scatter(x_target[0], y_target[0], z_target[0], color='red', label='Target Start')
# Intersection #
if len(solution.t_events[0]) > 0:
    ax.scatter(x_pursuer[-1], y_pursuer[-1], z_pursuer[-1], marker='x', color='green', label='Intersection')
else:
    ax.scatter(x_pursuer[-1], y_pursuer[-1], z_pursuer[-1], marker='x', color='blue', label='Pursuer End')
    ax.scatter(x_target[-1], y_target[-1], z_pursuer[-1], marker='x', color='red', label='Target End')
# Plot details #
ax.set_title('3D Pursuit Curve with Arclength')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.grid(True)
plt.show()