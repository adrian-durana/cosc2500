import numpy as np
import scipy as sp
## Solve for n = 2, 5, 10

## Hilbert matrix (A)
def hilbert(n):
    return np.fromfunction(lambda i, j: (1/(i+j+1)), (n, n), dtype=int)
    # Formula has been modified due to NumPy matrix indexing

## Ones column vector (x)
def ones(n):
    return np.ones(n)

## New ones column (x_c)
def x_vector(n):
    b_vector = np.matmul(hilbert(n), ones(n)) # b-vector
    return sp.linalg.solve(hilbert(n), b_vector) # x_c vector (LU-factorisation)

## Forward error
def forward_error(n):
    return np.linalg.norm(np.subtract(ones(n), x_vector(n)), np.inf) # forward error

## Error magnification factor
def error_mag(n):
    return (forward_error(n) / np.linalg.norm(ones(n), np.inf)) / (np.finfo(float).eps)

## Compare with condition number of A
def cond_one(n):
    return np.linalg.cond(hilbert(n), np.inf)

## Definition of condition number of A
def cond_two(n):
    return (np.linalg.norm(hilbert(n), np.inf) * np.linalg.norm(np.linalg.inv(hilbert(n)), np.inf))

#print(cond_one(2))
#print(cond_two(2))
#print(error_mag(2))
# error \approx condition number (see backward difference)

## Gaussian elimination (source: Blackboard, translated)
def gauss(A,B):
    if np.linalg.det(A) == 0:
        print("This system is unsolveable, as det(A) = 0.")
    b = B.reshape(-1, 1)
    a = np.hstack((A, b))
    n = b.shape[0]
    x = np.zeros(n)
    for j in range(n-1):
        if np.abs(a[j,j]) < np.finfo(float).eps:
            raise Exception("Zero pivor encountered")
        for i in range(j+1, n):
            mult = (a[i,j])/(a[j,j])
            for k in range(j+1, n):
                a[i,k] = a[i,k] - mult*(a[j,k])
            b[i] = b[i] - mult*(b[j])
    for i in range(n-1, -1, -1): # Values changed
        for j in range(i+1, n):
            b[i] = b[i] - a[i,j] * x[j]
        x[i] = b[i] / a[i,i]
    return x

a = 5
print(x_vector(a))
print(gauss(hilbert(a), (np.matmul(hilbert(a), ones(a)))))

## Part c
def matrix_two(n):
    return np.fromfunction(lambda i, j: (np.abs(i-j)+1), (n,n), dtype=int)
    # n = 100, 200, 300, 400, 500

## Part d
## For what values of n do the solutions in the above problems have no correct significant digits?
