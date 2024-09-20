import numpy as np

# Generates a function with a narrow spike, and some other interesting features.
def spikey(x):
    small_number = 1e-8
    denominator = (np.sqrt(3330)+x)**(2) + small_number
    return 1 + small_number - np.cos(x) + (1 / denominator)

# Maximum of fraction
maximum = 1 + 1e-8 - np.cos(np.sqrt(3330)) + 1e8

# Newton's method
def newton(x, tol=1e-9, maxiterations = 1000): # input: x (starting guess)
    h = 6.06*10**(-6)
    for i in range(maxiterations):
        numder_one = ((spikey(x+h) - spikey(x-h)) / (2*h))
        numder_two = (spikey(x+h) + spikey(x-h) - 2*spikey(x)) / (h**2)
        y = x - (numder_one / numder_two)
        if np.abs(x-y) <= tol: break
        x = y
    print('The approximate extremum is ' + str(x) + '.')
    print('The value of the function is ' + str(spikey(x)) + '.')
    print('The number of iterations required is ' + str(i+1) + '.')
