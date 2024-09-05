import numpy as np
import scipy as sp

# Hilbert matrix
def hilbert(n):
    matrix = np.fromfunction(lambda i, j: (1/(i+j+1)), (n, n), dtype=int)
    # Note that you have to change the formula from (1/i+j-1), due to numpy indexing!
    return matrix

# Identity column vector 
def ones(n):
    vector = np.ones((n,1))
    return vector

# b-vector
def bvector(n):
    answer = np.matmul(hilbert(n), ones(n))
    return answer

# Solve for x_c
def solve(n):
    solution = sp.linalg.solve(hilbert(n), bvector(n))
    return solution

# Solve for 2, 5, 10
print(solve(10))
