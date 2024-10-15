import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####################
## Laplace solver ##
####################

# A translation of poisson.m given in Sauer
# Finite difference sovler for 2D Poisson equation with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr] x [yb,yt] with M x N space steps
# Output: matrix 'w' holding solution values

def poisson(F,M=10,N=10,xl=0,xr=1,yb=0,yt=1):
    f = lambda x,y: 0
    if F == 1:
        g1 = lambda x: np.sin(np.pi*x) # 0 <= x <= 1
        g2 = lambda x: np.exp(-np.pi)*np.sin(np.pi*x) # 0 <= x <= 1
        g3 = lambda y: 0 # 0 <= y <= 1
        g4 = lambda y: 0 # 0 <= y <= 1
    if F == 2:
        g1 = lambda x: 0
        g2 = lambda x: 0
        g3 = lambda y: 0
        g4 = lambda y: np.sinh(np.pi)*np.sin(np.pi*y)

    m, n = M + 1, N + 1
    h = (xr-xl)/M
    k = (yt-yb)/N
    x = np.linspace(xl, xr, m)
    y = np.linspace(yb, yt, n)
    A = np.zeros((m*n,m*n))
    b = np.zeros((m*n,1))

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = i + j * m
            A[temp, temp] = -2 * (1 / h**2 + 1 / k**2)
            A[temp, temp - 1] = 1 / h**2
            A[temp, temp + 1] = 1 / h**2
            A[temp, temp - m] = 1 / k**2
            A[temp, temp + m] = 1 / k**2
            b[temp] = f(x[i], y[j])

    for i in range(m):
        A[i, i] = 1
        b[i] = g1(x[i])  
        temp = i + (n - 1) * m
        A[temp, temp] = 1
        b[temp] = g2(x[i])  
    
    for j in range(1, n - 1):
        temp = j * m 
        A[temp, temp] = 1
        b[temp] = g3(y[j]) 
        temp = (m - 1) + j * m 
        A[temp, temp] = 1
        b[temp] = g4(y[j]) 

    v = np.linalg.solve(A, b).flatten()
    Z = np.reshape(v, (m,n))

    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_xlim([xl, xr])
    ax.set_ylim([yb, yt])
    ax.view_init(30, 45)
    plt.show()

## True value
def true(F):
    def f(x, y):
        if F == 1: return np.exp(-np.pi*y)*np.sin(x*np.pi)
        if F == 2: return np.sinh(x*np.pi)*np.sin(y*np.pi)
    x = np.linspace(0, 1, num=100)
    y = np.linspace(0, 1, num=100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.view_init(30, 45)
    plt.show()

## Call function here
poisson(2)
true(2)