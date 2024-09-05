import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

# data arrays
x_values = np.array([
    0, 0.1341, 0.2693, 0.403, 0.5386, 0.6727, 
    0.8079, 0.9421, 1.0767, 1.2114, 1.3460, 1.4801,
    1.6153, 1.7495, 1.8847, 2.0199, 2.1540, 2.2886,
    2.4233, 2.5579, 2.6921, 2.8273, 2.9614, 3.0966,
    3.2307, 3.3659, 3.5000])
y_values = np.array([ 
    0, 0.0310, 0.1588, 0.3767, 0.6452, 0.8780, 
    0.9719, 1.0000, 0.9918, 0.9329, 0.8198, 0.7707, 
    0.8024, 0.7674, 0.6876, 0.5937, 0.5778, 0.4755, 
    0.3990, 0.3733, 0.2870, 0.2156, 0.2239, 0.1314, 
    0.1180, 0.0707, 0.0259])

# fit measure
n = len(x_values)
degree = []
fit_measure = []
for m in range(2,13):
    poly_fit = np.polyfit(x_values,y_values,m)
    poly_val = np.polyval(poly_fit, x_values)
    fit = np.sum((y_values - poly_val)**2) / (n - 2*m)
    degree.append(m)
    fit_measure.append(fit)

# fit measure plot
plt.plot(degree, fit_measure, '--bo', label = "fit measure")
plt.title("Fit measure vs. degree of polynomial")
plt.ylabel("fit measure")
plt.xlabel("degree of polynomial")
plt.legend()
plt.show()

# polynomial fitting
x_new = np.linspace(0, 3.5, num=100)
def polly(degree):
    Poly_fit = np.polyfit(x_values,y_values,degree)
    Poly_val = np.polyval(Poly_fit, x_values)
    y_new = np.polyval(Poly_fit,x_new)
    return y_new

# polynomial plotting
plt.plot(x_values,y_values, 'kD')
plt.plot(x_new, polly(11), 'y', label="degree 11")
plt.title("Polynomial fit, degree 11")
plt.legend()
plt.show()