import numpy as np
import scipy as sp

import time
start_time = time.time()

# n-sphere, centre at [0], length |1|

# Estimation of volume (rough code)
def estimate1(dim, n=1e6, i=0):
    for j in range(0, int(n)):
        x = 2*np.random.rand(dim) - 1
        if np.linalg.norm(x) <= 1: i += 1
    ratio = i/n
    return ratio

# Estimation of volume (more efficient)
def estimate(dim, n=int(1e8)):
    x = 2*np.random.rand(n, dim) - 1 # All random points generated at once
    distance = np.sum(x**2, axis=1) # Compute distance squared
    i = np.sum(distance <= 1)  # Vectorised array operations in NumPy are faster
    ratio = i / n
    return ratio

# Analytical solution of volume
def volume(dim):
    v = np.pi**(dim/2)/(sp.special.gamma(1 + (dim/2))*(2**dim))
    return v

# Results
for i in range(2,11):
    print('Dimension ' + str(i))
    print(estimate(i))
    print(volume(i))

print("Process finished --- %s seconds ---" % (time.time() - start_time))



## Create statistical estimate of result

0.78538845
0.7853981633974483