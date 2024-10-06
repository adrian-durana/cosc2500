import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def heatfd(xl, xr, yb, yt, M, N):
    # Define functions
    f = lambda x: np.sin(2 * np.pi * x)**2
    l = lambda t: np.zeros_like(t)
    r = lambda t: np.zeros_like(t)

    D = 1  # Diffusion coefficient
    h = (xr - xl) / M
    k = (yt - yb) / N
    m = M - 1
    n = N
    sigma = D * k / (h * h)

    # Define matrix a
    a = np.diag(1 - 2 * sigma * np.ones(m)) + np.diag(sigma * np.ones(m - 1), 1)
    a += np.diag(sigma * np.ones(m - 1), -1)

    # Boundary conditions
    lside = l(yb + np.arange(n + 1) * k)
    rside = r(yb + np.arange(n + 1) * k)

    # Initialize w with initial conditions
    w = np.zeros((m, n + 1))
    w[:, 0] = f(xl + np.arange(1, m + 1) * h)

    # Time-stepping loop
    for j in range(n):
        w[:, j + 1] = a @ w[:, j] + sigma * np.concatenate(([lside[j]], np.zeros(m - 2), [rside[j]]))

    # Add boundary conditions to the solution
    w = np.vstack([lside, w, rside])

    # Generate meshgrid for plotting
    x = np.linspace(xl, xr, m + 2)
    t = np.linspace(yb, yt, n + 1)

    # Plot the solution w using 3D surface plot
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, w.T, cmap='viridis')
    ax.view_init(60, 30)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('w')
    ax.set_xlim([xl, xr])
    ax.set_ylim([yb, yt])
    ax.set_zlim([-1, 1])
    plt.show()

# Example usage
xl = 0
xr = 1
yb = 0
yt = 1
M = 20
N = 50

heatfd(0.1, xr, yb, yt, M, N)
