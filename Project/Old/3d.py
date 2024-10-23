import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

## Parameters
k = 2
x0, y0, z0 = 5, 0, 1
tmax = 5
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory
def T_x(t):
    return 0*t
def T_y(t):
    return t
def T_z(t):
    return 0*t + 1
def T(t):
    return np.array([T_x(t), T_y(t), T_z(t)])

## Pursuer trajectory - system of ODEs 
def pursuit_curve(t, P):
    x, y, z = P
    diff = T(t) - np.array([x, y, z]) # Distance vector
    norm_diff = np.linalg.norm(diff) # Norm of vector
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1], direction[2]]

## ODE solution
solution = solve_ivp(pursuit_curve, [0, tmax], [x0, y0, z0], t_eval=t_eval)
x_pursuer = solution.y[0]
y_pursuer = solution.y[1]
z_pursuer = solution.y[2]
x_target = T(t_eval)[0]
y_target = T(t_eval)[1]
z_target = T(t_eval)[2]

## Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Pursuer #
ax.plot(x_pursuer, y_pursuer, z_pursuer, label='Pursuer', color='blue')
ax.scatter(x_pursuer[0], y_pursuer[0], z_pursuer[0], color='blue', label='Pursuer Start')
# Target #
ax.plot(x_target, y_target, z_target, label='Target', color='red')
ax.scatter(x_target[0], y_target[0], z_target[0], color='red', label='Target Start')
# Plot details #
ax.set_title('3D Pursuit Curve')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.grid(True)
plt.show()