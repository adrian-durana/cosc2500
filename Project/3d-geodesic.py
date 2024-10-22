import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Surface
def surface_z(x, y):
    return x**2 + y**2

# Gradient of the surface: dz/dx, dz/dy
def surface_gradient(x, y):
    dz_dx = 2*x
    dz_dy = 2*y
    return dz_dx, dz_dy

# Parameters
k = 1  # Pursuer speed factor
h = 0.01
x0, y0, z0 = -1, 0, 1  # Initial position of the pursuer
initial_conditions = [x0, y0, z0]
tmax = 2

# Time span for the simulation
t_span = (0, tmax)
t_eval = np.linspace(t_span[0], t_span[1], int(1/h))

# Target trajectory constrained to the surface
def T_x(t):
    return t
def T_y(t):
    return t
def T_z(t):
    return surface_z(T_x(t), T_y(t))

def T(t):
    return np.array([T_x(t), T_y(t), T_z(t)])

# Geodesic equations for the surface
# This governs how the pursuer moves on the surface.
def geodesic_equations(t, P):
    x, y, dx, dy = P
    
    # Calculate surface gradient (dz/dx, dz/dy)
    dz_dx, dz_dy = surface_gradient(x, y)
    
    # Christoffel symbols for the paraboloid z = x^2 + y^2
    Γxx = 2*x / (1 + 4*x**2 + 4*y**2)
    Γxy = 2*y / (1 + 4*x**2 + 4*y**2)
    
    # Geodesic equations (second-order ODEs)
    ddx_dt = -Γxx * dx**2 - 2*Γxy * dx * dy
    ddy_dt = -Γxy * dy**2 - 2*Γxx * dx * dy

    return [dx, dy, ddx_dt, ddy_dt]

# Pursuer's ODEs: follows geodesics while pursuing the target
def pursuit_geodesic(t, P):
    x, y, z = P  # Current position of the pursuer
    # Distance vector from pursuer to target (projected in x, y plane)
    tx, ty, tz = T(t)
    diff = np.array([tx - x, ty - y, 0])
    
    # Unit vector toward the target
    norm_diff = np.linalg.norm(diff[:2])
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    
    # Update z to remain on the surface
    z = surface_z(x, y)
    
    # Move in the geodesic direction (while aiming at the target)
    dx, dy = direction[0], direction[1]
    
    return [dx, dy, z - surface_z(x, y)]

# Initial conditions for the pursuer (initial velocity is 0, no motion)
pursuer_initial_conditions = [x0, y0, z0]

# Solve the system for the pursuer's path
pursuer_solution = solve_ivp(pursuit_geodesic, t_span, pursuer_initial_conditions, t_eval=t_eval)
x_pursuer = pursuer_solution.y[0]
y_pursuer = pursuer_solution.y[1]
z_pursuer = surface_z(x_pursuer, y_pursuer)  # Ensure z stays on the surface

# Target's trajectory
x_target = T_x(t_eval)
y_target = T_y(t_eval)
z_target = T_z(t_eval)

# Plotting the results
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Surface #
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = surface_z(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plot pursuer's trajectory
ax.plot(x_pursuer, y_pursuer, z_pursuer, label='Pursuer (Geodesic)', color='blue')
ax.scatter(x_pursuer[0], y_pursuer[0], z_pursuer[0], color='blue', label='Pursuer Start')

# Plot target's trajectory
ax.plot(x_target, y_target, z_target, label='Target', color='red')
ax.scatter(x_target[0], y_target[0], z_target[0], color='red', label='Target Start')

# Surface plot
ax.set_title('3D Pursuit Curve with Geodesics on a Surface')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.grid(True)
plt.show()
