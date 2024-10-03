import numpy as np

## Euler-Maruyama method
def euler_maruyama(h):
    n = int(1/h)
    W = np.zeros(n+1) # y(0) = 0
    rand = np.random.normal(0,1,n)
    dB = B = np.zeros(n)
    for i in range(n):
        dB[i] = rand[i] * np.sqrt(h)
        B[i] = np.sum(dB)
        W[i+1] = W[i] + B[i]*h + (9*W[i]**2)**(1/3)
    return W[-2]

## Perform 5000 simulations
def group(h, n=5000):
    A = np.zeros(n)
    for i in range(n): A[i] = euler_maruyama(h)
    print(np.mean(A))

## Call the function here!
group(1e-1)

## How to find the error