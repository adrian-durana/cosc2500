import numpy as np

def gauss(A, B):
    if np.linalg.det(A) == 0:
        print("This system is unsolvable, as det(A) = 0.")
        return None
    
    # Reshape B into a column vector and create the augmented matrix
    b = B.reshape(-1, 1)
    a = np.hstack((A, b))
    print("Augmented Matrix:\n", a)

    # Number of rows
    n = a.shape[0]
    x = np.zeros(n)
    
    # Forward elimination
    for j in range(n - 1):  # Corrected to range(0, n-1)
        if np.abs(a[j, j]) < np.finfo(float).eps:
            raise Exception("Zero pivot encountered")
        for i in range(j + 1, n):
            mult = a[i, j] / a[j, j]
            for k in range(j, n + 1):  # Includes the last column (b values)
                a[i, k] -= mult * a[j, k]
    
    # Back substitution
    for i in range(n - 1, -1, -1):  # Corrected to range(n-1, -1, -1) to match MATLAB loop
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += a[i, j] * x[j]
        x[i] = (a[i, -1] - sum_ax) / a[i, i]  # Last column of `a` is `b`

    return x

# Example usage
A = np.array([[1, 0.5], [0.5, 1/3]])
B = np.array([1.5, 5/6])
solution = gauss(A, B)
print("Solution vector x:", solution)
