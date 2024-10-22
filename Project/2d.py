import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

## Parameters
k = 1
h = 0.001
x0, y0 = 5, 0
initial_conditions = [x0, y0]
tmax = 10

## Time span for the simulation
t_span = (0, tmax)
t_eval = np.linspace(t_span[0], t_span[1], int(1/h))

## Target trajectory
def T_x(t):
    return 0*t
def T_y(t):
    return t
def T(t):
    return np.array([T_x(t), T_y(t)])

## Pursuer trajectory - system of ODEs
def pursuit_curve(t, P):
    x, y = P
    # Distance vector from pursuer to target
    diff = T(t) - np.array([x, y])
    # Normalized direction of the target
    norm_diff = np.linalg.norm(diff)
    direction = k * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1]]

## ODE solution
solution = solve_ivp(pursuit_curve, t_span, initial_conditions, t_eval=t_eval)
x_pursuer = solution.y[0]
y_pursuer = solution.y[1]
x_target = T(t_eval)[0]
y_target = T(t_eval)[1]

## Real solution
x_span = np.linspace(0.001, 5, int(1/0.001)-1)
def true(x, x0=x0, y0=y0):
    eta = (x / x0)**2
    r0 = np.sqrt(x0**2 + y0**2)
    chi = ((r0+y0)/(r0-y0))
    y = (1/4)*((y0+r0)*eta + (y0-r0)*np.log(eta) + 3*y0 - r0)
    return y

## Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_span, true(x_span), color='green') ## Real solution
plt.plot(x_pursuer, y_pursuer, label='Pursuer', color='blue')
plt.scatter(x_pursuer[0], y_pursuer[0], color='blue')
plt.plot(x_target, y_target, label='Target', color='red')
plt.scatter(x_target[0], y_target[0], color='red')
plt.title('Pursuit Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()