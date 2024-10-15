import numpy as np
import matplotlib.pyplot as plt

##########################
## Wave equation solver ##
##########################

# Uses explanation given in Sauer
# Input: space interval [xl,xr], time interval [yb,yt], space and time steps M, N
# Output: matrix 'w' holding solution values

def wave(F, xl=0, xr=1, yb=0, yt=1, h=0.05):
    # Functions
    if F == 1:
        c = 4
        f = lambda x: np.sin(np.pi * x) 
        g = lambda x: 0 
        l = lambda t: 0
        r = lambda t: 0
    if F == 2:
        c = 2
        f = lambda x: np.exp(-x)
        g = lambda x: -2*np.exp(-x)
        l = lambda t: np.exp(-2*t)
        r = lambda t: np.exp(-1-2*t) 
    k = h/c
    # Variables
    M = int((xr - xl) / h) + 1  # x-points
    N = int((yt - yb) / k) + 1  # y-points
    sigma = (c * k / h)
    s2 = sigma**2

    # Arrays (i, j) (space, time)
    x = np.linspace(xl, xr, M)
    t = np.linspace(yb, yt, N)
    u = np.zeros((M, N))      

    # Initial and boundary conditions
    u[:, 0] = f(x)                
    u[:, 1] = u[:, 0] + k * g(x) # Uses initial derivative
    u[0, :] = l(t) 
    u[-1, :] = r(t) 

    # Finite difference
    for j in range(1, N - 1):
        for i in range(1, M - 1):
            u[i, j+1] = (2*(1 - s2)*u[i, j] - u[i, j-1] + s2*(u[i+1, j] + u[i-1, j]))
        u[0, j+1] = l(t[j+1])
        u[-1, j+1] = r(t[j+1])

    # Plot
    X, T = np.meshgrid(x, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(x,t)')
    ax.view_init(30, 45)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()

## True value
def true(F):
    def f(x, t, F=F):
        if F == 1: return np.sin(np.pi*x)*np.cos(4*np.pi*t)
        if F == 2: return np.exp(-x-2*t)
    x = np.linspace(0, 1, num=1000)
    y = np.linspace(0, 1, num=1000)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(x,t)')
    ax.view_init(30, 45)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()

## Call function here
wave(2)
true(2)