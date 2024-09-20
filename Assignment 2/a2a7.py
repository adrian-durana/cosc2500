import numpy as np

def function(x,y):
    return -x**4 + x**2 + y**2
    #return -(x**2 + y**2)
    #return x**4 - x**2 + y**2
    #return x**2 + y**2
    #return 5*x**4 + 4*y*x**2 - x*y**3 + 4*y**4 - x

def gradient(x,y):
    h = 6.06*10**(-6)
    derivative_x = (function(x+h, y) - function(x-h, y)) / (2*h)
    derivative_y = (function(x, y+h) - function(x, y-h)) / (2*h)
    gradient = np.array([derivative_x, derivative_y])
    return gradient

def line_search(x, y, maxiterations=1000, s=0, step_size=0.1):
    [v_x, v_y] = gradient(x,y)
    for i in range(maxiterations):
        x_new = x - s * v_x
        y_new = y - s * v_y
        if function(x_new, y_new) < function(x, y): break
        s += step_size
    return s

def steepest_descent(x, y, tol=1e-6, maxiterations=1000):
    for i in range(maxiterations):
        gradient_norm = np.linalg.norm(gradient(x,y))
        if gradient_norm <= tol: break
        x = x - line_search(x,y)*gradient(x,y)[0]
        y = y - line_search(x,y)*gradient(x,y)[1]
    print("The approximate minimum is (" + str(x) + "," + str(y) + ")")
    print("The value of the function is: " + str(function(x,y)) + ".") 
    print("The number of iterations required is: " + str(i+1) + ".")
