import numpy as np
import matplotlib.pyplot as plt

## Euler-Maruyama method
def euler_maruyama(h):
    n = int(1/h)
    W = np.zeros(n+1) 
    rand = np.random.normal(0,1,n)
    dB = np.zeros(n+1)
    B = np.zeros(n+1)
    dB[0] = rand[0] * np.sqrt(h)
    for i in range(n):
        dB[i+1] = rand[i] * np.sqrt(h)
        B[i+1] = B[i] + dB[i+1]
        W[i+1] = W[i] + B[i]*h + (9*W[i]**2)**(1/3)*h
    return W #W[-2]

## Perform 5000 simulations
def group(h, n=5000):
    A = np.zeros(n)
    for i in range(n): A[i] = euler_maruyama(h)
    print(np.mean(A))

## How to find the error
def plot(h):
    n = int(1/h)
    x_values = np.linspace(0,1,num=(n+1))
    plt.plot(x_values, euler_maruyama(h))
    plt.show()

## Call the function here!
group(0.001)
plot(0.01)

