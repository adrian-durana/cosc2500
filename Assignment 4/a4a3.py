import numpy as np

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
    return W[-1]
    
## Perform 5000 simulations
def group(h, n=5000):
    A = np.zeros(n+1)
    for i in range(n): A[i] = euler_maruyama(h)
    print(np.mean(A))

## Call the function here!
group(0.1)
group(0.01)
group(0.001)
group(0.0001)