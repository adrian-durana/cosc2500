import numpy as np
import scipy as sp

# n-sphere, centre at [0], length |1|

# Estimation of volume (rough code)
def estimate1(dim, n=1e6, i=0):
    for j in range(0, int(n)):
        x = 2*np.random.rand(dim) - 1
        if np.linalg.norm(x) <= 1: i += 1
    ratio = i/n
    return ratio

# Estimation of volume (more efficient)
def estimate(dim, n=int(1e9)):
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

"""
dimension2
0.784955
0.7853981633974483
dimension3
0.522927
0.5235987755982989
dimension4
0.309237
0.30842513753404244
dimension5
0.163929
0.16449340668482262
dimension6
0.080904
0.08074551218828077
dimension7
0.036423
0.03691223414321407
dimension8
0.015741
0.0158543442438155
dimension9
0.006552
0.006442400200661536
dimension10
0.002509
0.00249039457019272
"""

## Create statistical estimate of result

0.78538845
0.7853981633974483