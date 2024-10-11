import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def two_body_orbit(inter, ic, n, p):
    ## Constants and vectors
    G = 1  # Gravitational constant
    m1 = 0.03  # Mass of body 1
    m2 = 0.3   # Mass of body 2
    h = (inter[1] - inter[0]) / n  # Time step size
    y = np.zeros((n + 1, 8))  # State vector for positions and velocities
    y[0, :] = ic  # Initial conditions
    t = np.linspace(inter[0], inter[1], n + 1)  # Time vector

    ## Plot setup
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(-5, 5, 1))
    ax.set_yticks(np.arange(-5, 5, 1))
    body1_line, = ax.plot([], [], 'bo', markersize=8)  # Body 1 (blue)
    body2_line, = ax.plot([], [], 'ro', markersize=8)  # Body 2 (red)
    body1_trace, = ax.plot([], [], 'b-', alpha=0.5)  # Trace for Body 1
    body2_trace, = ax.plot([], [], 'r-', alpha=0.5)  # Trace for Body 2

    def ydot(t, x):
        px1, vx1, py1, vy1, px2, vx2, py2, vy2 = x # Extract positions and velocities
        dist = np.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2) # Calculate distances
        z = np.zeros(8) # Derivative vector
        z[0] = vx1  # dx1/dt
        z[1] = G * m2 * (px2 - px1) / (dist ** 3)  # dvx1/dt
        z[2] = vy1  # dy1/dt
        z[3] = G * m2 * (py2 - py1) / (dist ** 3)  # dvy1/dt 
        z[4] = vx2  # dx2/dt
        z[5] = -G * m1 * (px2 - px1) / (dist ** 3)  # -dvx2/dt
        z[6] = vy2  # dy2/dt
        z[7] = -G * m1 * (py2 - py1) / (dist ** 3)  # -dvy2/dt 
        return z

    def eulerstep(t, x, h):
        return x + h * ydot(t, x)

    for k in range(n // p):
        for i in range(p):
            y[i+1, :] = eulerstep(t[i], y[i, :], h)  # Update positions and velocities
        y[0, :] = y[p, :]
        body1_line.set_data([y[0, 0]], [y[0, 2]])  # Update position of Body 1
        body2_line.set_data([y[0, 4]], [y[0, 6]])  # Update position of Body 2
        plt.pause(0.01)  # Pause to update the plot
    plt.show()

two_body_orbit([0, 100], [2, 0.2, 2, -0.2, 0, -0.02, 0, 0.02], 1000, 10)