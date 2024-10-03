import numpy as np

a = -2 # or -5 or -8
b =  5 # or 3 or 3

## Singular walk
def walk(a, b, p = 0.7, max = int(1e6), w=0, j=0):
    R = np.random.rand(max)
    for i in range(max):
      if R[i] < p: w += 1
      else: w -= 1
      if w == a:
        break
      if w == b:
        j = 1
        break
    return j

## Compilation of n walks
def group(a, b, n=10000):
    A = []
    for i in range(n):
        A.append(walk(a, b))
    ratio = np.sum(A) / n
    print(ratio)

## True value of probability
def true(a,b,p=0.7):
    true = (((1-p)/p)**(-a) - 1)/(((1-p)/p)**(b-a) - 1)
    return true

## Call functions here
print(group(-2, 5))
print(true(-2, 5))

#(-2,5) or (-5,3) or (-8,3)
