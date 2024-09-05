import numpy as np
import scipy as sp

## Hilbert matrix
def hilbert(n):
    matrix = np.fromfunction(lambda i, j: (1/(i+j+1)), (n, n), dtype=int)
    # Note that you have to change the formula from (1/i+j-1), due to numpy indexing!
    return matrix

## Identity column vector 
def ones(n):
    vector = np.ones((n,1))
    return vector

## b-vector
def bvector(n):
    answer = np.matmul(hilbert(n), ones(n))
    return answer

## Solve for x_c
def solve(n):
    solution = sp.linalg.solve(hilbert(n), bvector(n))
    return solution

## Solve for 2, 5, 10
print(solve(10))

# Gaussian elimination (see Sauer, or Blackboard)

# Forward error - infnorm of difference between actual solution and computed solution ||x-x_a||_inf

np.linalg.norm([matrix], ord=inf)

# Error magnification factor - [ infnorm(x-x_c) / infnorm(x) ] / machine-eps.

# Compare with ccondition number of A


# Part c
def matrix_two(n):
    matrix = np.fromfunction(lambda i, j: (np.abs(i-j)+1), (n,n), dtype=int)
    # No need to change formula type!
    return matrix
    # n = 100, 200, 300, 400, 500

# Part d:
# For what values of n do the solutions in the above problems have no correct significant digits?




