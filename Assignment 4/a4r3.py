import numpy as np
import time
start_time = time.time()

## Singular walk
def walk(a, b, p = 0.7, max = int(1e6), w=0, j=0):
    steps = np.where(np.random.rand(max) < p, 1, -1)
    for i in range(max):
        w += steps[i]
        if w == a: break
        elif w == b: j = 1; break
    return j

## Compilation of n walks
def group(a, b, n=10000):
    A = np.zeros(n)
    for i in range(n): A[i] = walk(a,b)
    return np.mean(A)

## True value of probability
def true(a,b,p=0.7):
    true = (((1-p)/p)**(-a) - 1)/(((1-p)/p)**(b-a) - 1)
    return true

## Call functions here
print(group(-2, 5))
print(true(-2, 5))

## Coordinates: (-2,5) or (-5,3) or (-8,3)
print("--- Execution time: %s seconds ---" % (time.time() - start_time))

## Estimated run time: 190s
"""
0.8168
0.8185001387948708
--- Execution time: 190.39128708839417 seconds ---
"""