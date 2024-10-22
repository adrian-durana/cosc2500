import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Surface
def surface_z(x, y):
    return x**2 + y**2

## Parameters
k = 1 
h = 0.001
x0, y0 = -1, 0
initial_conditions = [x0, y0, surface_z(x0,y0)]
tmax = 2

## Time span for the simulation
t_span = (0, tmax)
t_eval = np.linspace(t_span[0], t_span[1], int(1/h))

## Target trajectory, constrained to surface
def T_x(t):
    return t
def T_y(t):
    return t
def T_z(t):
    return surface_z(T_x(t), T_y(t))
def T(t):
    return np.array([T_x(t), T_y(t), T_z(t)])

## Pursuer trajectory - system of ODEs 
def pursuit_curve(t, P):
    x, y, z = P
    # Distance vector from pursuer to target
    diff = T(t) - np.array([x, y, z])
    # Normalized direction of the target
    norm_diff = np.linalg.norm(diff)
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    z = surface_z(x, y) # z remains on surface
    return [direction[0], direction[1], direction[2]]

## ODE solution
solution = solve_ivp(pursuit_curve, t_span, initial_conditions, t_eval=t_eval)
x_pursuer = solution.y[0]
y_pursuer = solution.y[1]
z_pursuer = surface_z(x_pursuer, y_pursuer)  # Ensure z stays on the surface
x_target = T_x(t_eval)
y_target = T_y(t_eval)
z_target = T_z(t_eval)

## Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Surface #
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = surface_z(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
# Pursuer #
ax.plot(x_pursuer, y_pursuer, z_pursuer, label='Pursuer', color='blue')
ax.scatter(x_pursuer[0], y_pursuer[0], z_pursuer[0], color='blue', label='Pursuer Start')
# Target #
ax.plot(x_target, y_target, z_target, label='Target', color='red')
ax.scatter(x_target[0], y_target[0], z_target[0], color='red', label='Target Start')
# Plot details #
ax.set_title('3D Pursuit Curve on a Surface')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.grid(True)
plt.show()
