import numpy as np

## Euler-Maruyama method

### Calculate the number of steps ###
a = 0
b = 1
h = 1e-1 ## 0.1, 0.01, 0.001
n = int((b-a) / h)

### Perform 5000 simulations ###


### Euler-Maruyama step ###
w = np.zeros(n+1)
rand = np.random.normal(0,1,n)
dB = np.zeros(n)
B_t = np.zeros(n)
w[0] = 0 # y(0) = 0

for i in range(n):
    dB[i] = rand[i] * np.sqrt(h)
    B_t[i] = np.sum(dB)
    w[i+1] = w[i] + B_t[i]*h + (9*w[i]**2)**(1/3)

print(w[-1])


### Find the mean ###


