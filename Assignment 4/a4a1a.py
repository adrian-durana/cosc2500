import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##########################
## Heat equation solver ##
##########################

# A translation of heatfd.m given in Sauer
# Forward difference method for heat equation
# Input: space interval [xl,xr], time interval [yb,yt], space and time steps M, N
# Output: matrix 'w' holding solution values

def heatfd(F,N=500,xl=0,xr=1,yb=0,yt=1,M=10):
    D = 2
    if F == 1:
        f = lambda x: 2*np.cosh(x)
        l = lambda t: 2*np.exp(2*t)
        r = lambda t: (np.exp(2)+1)*np.exp(2*t-1)
    if F == 2:
        f = lambda x: np.exp(x)
        l = lambda t: np.exp(2*t)
        r = lambda t: np.exp(2*t+1)
    
    m, n = M-1, N
    h = (xr-xl)/M
    k = (yt-yb)/N
    sigma = D*k / (h*h)

    a = np.diag(1 - 2*sigma*np.ones(m)) + np.diag(sigma*np.ones(m-1), 1)
    a += np.diag(sigma*np.ones(m-1), -1) 

    lside = np.vectorize(l)(yb + np.arange(n+1)*k)
    rside = np.vectorize(r)(yb + np.arange(n+1)*k)

    w = np.zeros((m,n+1))
    w[:, 0] = f(xl + np.arange(1, m+1)*h)
    
    for j in range(n):
        w[:, j+1] = a @ w[:, j] + sigma * np.concatenate(([lside[j]], np.zeros(m-2), [rside[j]]))

    Z = np.transpose(np.vstack([lside,w,rside]))
    x = np.linspace(xl, xr, m+2)
    t = np.linspace(yb, yt, n+1)

    X, T = np.meshgrid(x,t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f(x,t)')
    ax.view_init(30, -45)
    ax.set_xlim([xl, xr])
    ax.set_ylim([yb, yt])
    plt.show()

## True value
def true(F):
    def f(x, t, F=F):
        if F == 1: return np.exp(2*t+x) + np.exp(2*t-x)
        if F == 2: return np.exp(2*t+x)
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
    ax.view_init(30, -45)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()

## Call function here
heatfd(1,382)
heatfd(2,382)
true(2)