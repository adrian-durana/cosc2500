import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Function
F = 1

######################
### Poisson solver ###
######################

# A translation of poisson.m given in Sauer
# Finite difference sovler for 2D Poisson equation with Dirichlet boundary conditions on a rectangle
# Input: rectangle domain [xl,xr] x [yb,yt] with M x N space steps
# Output: matrix 'w' holding solution values

def poisson(xl,xr,yb,yt,M,N):
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
    Z = np.transpose(np.reshape(v, (m,n)))

    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()

## True value
def true():
    def f(x, t):
        if F == 1: return np.exp(2*t+x) + np.exp(2*t-x)
        if F == 2: return np.exp(2*t+x)
    x = np.linspace(0, 1, num=30)
    y = np.linspace(0, 1, num=30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()

poisson(0,1,0,1,100,100)
true()