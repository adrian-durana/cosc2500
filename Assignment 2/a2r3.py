import numpy as np

## Part a
# def x_vector(n): ## Computed solution of x (x_c)
#     b_vector = np.matmul(sp.linalg.hilbert(n), np.ones(n)) # b-vector
#     return np.linalg.solve(sp.linalg.hilbert(n), b_vector) # x_c vector (LU-factorisation)
# def forward_error(n): ## Forward error
#     return np.linalg.norm(np.ones(n) -  x_vector(n), np.inf) # forward error
# def error_mag(n): ## Error magnification factor
#     return (forward_error(n) / np.linalg.norm(np.ones(n), np.inf)) / (np.finfo(float).eps)
# def cond(n): ## Compare with condition number of A
#     return np.linalg.cond(sp.linalg.hilbert(n), np.inf)

## Part b
def matrix_two(n):
    return np.fromfunction(lambda i, j: (np.abs(i-j)+1), (n,n), dtype=int)
def x_vector(n): ## Computed solution of x (x_c)
    b_vector = np.matmul(matrix_two(n), np.ones(n)) # b-vector
    return np.linalg.solve(matrix_two(n), b_vector) # x_c vector (LU-factorisation)
def forward_error(n): ## Forward error
    return np.linalg.norm(np.ones(n) -  x_vector(n), np.inf) # forward error
def error_mag(n): ## Error magnification factor
    return (forward_error(n) / np.linalg.norm(np.ones(n), np.inf)) / (np.finfo(float).eps)
def cond(n): ## Compare with condition number of A
    return np.linalg.cond(matrix_two(n), np.inf)