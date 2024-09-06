import numpy as np
import scipy as sp

## Hilbert matrix (A)
def hilbert(n):
    return np.fromfunction(lambda i, j: (1/(i+j+1)), (n, n), dtype=int)
    # Formula has been modified due to NumPy matrix indexing

## Ones column vector (x)
def ones(n):
    return np.ones((n,1))

## Forward error
def forward_error(n):
    b_vector = np.matmul(hilbert(n), ones(n)) # b-vector
    x_vector = sp.linalg.solve(hilbert(n), b_vector) # x_c vector
    return np.linalg.norm(np.subtract(ones(n), x_vector), np.inf) # forward error

## Error magnification factor
def error_mag(n):
    return (forward_error(n) - np.linalg.norm(ones(n), np.inf)) / (np.finfo(float).eps)

## Compare with condition number of A
def cond(n):
    return np.linalg.cond(hilbert(n), np.inf)

## Solve for 2, 5, 10
print(hilbert(2))


# Gaussian elimination (see Sauer, or Blackboard)

# Part c
def matrix_two(n):
    matrix = np.fromfunction(lambda i, j: (np.abs(i-j)+1), (n,n), dtype=int)
    # No need to change formula type!
    return matrix
    # n = 100, 200, 300, 400, 500


# Part d:
# For what values of n do the solutions in the above problems have no correct significant digits?
