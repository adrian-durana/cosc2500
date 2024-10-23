import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Surface
def surface_z(x, y):
    return np.sin(x)*np.cos(y)

## Parameters
k = 1
x0, y0 = 5, 0
tmax = 3
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory, constrained to surface
def T_x(t):return 0*t
def T_y(t): return t
def T_z(t): return surface_z(T_x(t), T_y(t))
def T(t): return np.array([T_x(t), T_y(t), T_z(t)])

## Pursuer trajectory - system of ODEs 
def pursuit_curve(t, P, k):
    x, y, z = P
    diff = T(t) - np.array([x, y, z])  # Distance vector
    norm_diff = np.linalg.norm(diff)  # Norm of vector
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1], direction[2]]

## Arclength calculation
def compute_arclength(x, y, z):
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    return np.sum(diffs)

# Store pursuer trajectory
pursuer_trajectory = np.zeros((len(t_eval), 3))
pursuer_trajectory[0] = [x0, y0, surface_z(x0, y0)]


# Loop through each time step to adjust k dynamically
for i in range(len(t_eval) - 1):
    # Solve the ODE for the current timestep with current k
    sol = solve_ivp(pursuit_curve, [t_eval[i], t_eval[i + 1]], pursuer_trajectory[i], args=(k,), max_step=h)
    
    # Update the pursuer's position for the next timestep
    pursuer_trajectory[i + 1] = sol.y[:, -1]  # Take the last point in the solution

# Extract pursuer's coordinates
x_pursuer = pursuer_trajectory[:, 0]
y_pursuer = pursuer_trajectory[:, 1]
z_pursuer = surface_z(x_pursuer, y_pursuer)  # Ensure z stays on the surface
x_target = T_x(t_eval)
y_target = T_y(t_eval)
z_target = T_z(t_eval)

# Calculate initial arclengths
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer, z_pursuer)
arclength_target = compute_arclength(x_target, y_target, z_target)

# Adjust k based on the ratio of arclengths
if arclength_target > 0:
    k_adjusted = arclength_target / arclength_pursuer
    print(k_adjusted)
    k *= k_adjusted

# Store pursuer trajectory again with new k
pursuer_trajectory = np.zeros((len(t_eval), 3))
pursuer_trajectory[0] = [x0, y0, surface_z(x0, y0)]

# Repeat the ODE solving with the adjusted k
for i in range(len(t_eval) - 1):
    # Solve the ODE for the current timestep with adjusted k
    sol = solve_ivp(pursuit_curve, [t_eval[i], t_eval[i + 1]], pursuer_trajectory[i], args=(k,), max_step=h)
    
    # Update the pursuer's position for the next timestep
    pursuer_trajectory[i + 1] = sol.y[:, -1]  # Take the last point in the solution

# Extract new pursuer's coordinates
x_pursuer = pursuer_trajectory[:, 0]
y_pursuer = pursuer_trajectory[:, 1]
z_pursuer = surface_z(x_pursuer, y_pursuer)

# Arclength calculation
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer, z_pursuer)
arclength_target = compute_arclength(x_target, y_target, z_target)
print(f"Adjusted Arclength of pursuer: {arclength_pursuer:.2f}")
print(f"Adjusted Arclength of target: {arclength_target:.2f}")

## Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
# Surface #
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = surface_z(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
# Pursuer
ax.plot(x_pursuer, y_pursuer, z_pursuer, label='Pursuer', color='blue')
ax.scatter(x_pursuer[0], y_pursuer[0], z_pursuer[0], color='blue', label='Pursuer Start')
# Target
ax.plot(x_target, y_target, z_target, label='Target', color='red')
ax.scatter(x_target[0], y_target[0], z_target[0], color='red', label='Target Start')

# Plot details
ax.set_title('3D Pursuit Curve on a Surface with Dynamic Speed Adjustment')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 10)
ax.legend()
ax.grid(True)
ax.view_init(90,0)
plt.show()