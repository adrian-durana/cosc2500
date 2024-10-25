import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Two-dimensional pursuit curve solver
With arclength to check lengths
"""

## Parameters
k = 1
x0, y0 = 0, 2
tmax = np.pi*16
h = 0.001
t_eval = np.linspace(0, tmax, int(1/h))

## Target trajectory
def T_x(t): return np.cos(t) #0*t # np.cos(2*t)*np.cos(t)
def T_y(t): return np.sin(t) #t # np.cos(2*t)*np.sin(t)
def T(t): return np.array([T_x(t), T_y(t)])

## Target derivative
def dT_x(t): return -np.sin(t) # 0*t # -(2*np.sin(2*t)*np.cos(t)+np.cos(2*t)*np.sin(t))
def dT_y(t): return np.cos(t) # 0*t + 1 # -2*np.sin(2*t)*np.sin(t)+np.cos(2*t)*np.cos(t)
def dT(t): return np.array([dT_x(t), dT_y(t)])

## Pursuer trajectory - system of ODEs
def pursuit_curve(t, P):
    x, y = P
    diff = T(t) - np.array([x, y])  # Distance vector
    norm_diff = np.linalg.norm(diff)  # Norm of vector
    direction = k * np.sqrt(np.sum(dT(t)**2)) * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1]]

## Event function
def proximity_event(t, P):
    x, y = P
    distance = np.linalg.norm(T(t) - np.array([x, y])) # Norm of distance vector 
    return distance - 0.01  # Trigger when distance is less than 1e-6
proximity_event.terminal = True  # Terminate integration if event occurs

## ODE solution
solution = solve_ivp(pursuit_curve, [0, tmax], [x0, y0], t_eval=t_eval, 
                     events=proximity_event)

## Event checker
if len(solution.t_events) > 0 and len(solution.t_events[0]) > 0:
    t_intersect = solution.t_events[0][0]
    mask = solution.t <= t_intersect # Mask all points up to intersection
else:
    mask = np.ones_like(solution.t, dtype=bool) # Mask all points

## Curve outputs
x_pursuer = solution.y[0][mask]
y_pursuer = solution.y[1][mask]
x_target = T(solution.t[mask])[0]
y_target = T(solution.t[mask])[1]

## Arclength calculation
def compute_arclength(x, y):
    diffs = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.sum(diffs)
arclength_pursuer = compute_arclength(x_pursuer, y_pursuer)
arclength_target = compute_arclength(x_target, y_target)
print(f"Arclength of pursuer: {arclength_pursuer:.6f}")
print(f"Arclength of target: {arclength_target:.6f}")

## Prediction: pursuer trajectory system
def pursuit_curve_with_prediction(t, P, time_ahead=1):
    x, y = P
   
    t_future = t + time_ahead # Future time
    #if t_future > tmax: t_future = tmax # Error handling: no predictions beyond tmax
    future_target_position = T(t_future)
    future_target_derivative = dT(t_future)
    
    diff = future_target_position - np.array([x, y])
    norm_diff = np.linalg.norm(diff)  # Norm of vector
    direction = k * np.sqrt(np.sum(future_target_derivative**2)) * diff / norm_diff if norm_diff != 0 else np.zeros_like(diff)
    return [direction[0], direction[1]]

## Prediction: ODE solution
solution_with_prediction = solve_ivp(pursuit_curve_with_prediction, [0, tmax], [x0, y0], t_eval=t_eval, 
                                     events=proximity_event)

## Prediction: Event checker 
if len(solution_with_prediction.t_events) > 0 and len(solution_with_prediction.t_events[0]) > 0:
    t_intersect_predicted = solution_with_prediction.t_events[0][0]
    mask_predicted = solution_with_prediction.t <= t_intersect_predicted
else:
    mask_predicted = np.ones_like(solution_with_prediction.t, dtype=bool)

## Prediction: Curve outputs
x_pursuer_predicted = solution_with_prediction.y[0][mask_predicted]
y_pursuer_predicted = solution_with_prediction.y[1][mask_predicted]
x_target_predicted = T(solution_with_prediction.t[mask_predicted])[0]
y_target_predicted = T(solution_with_prediction.t[mask_predicted])[1]

## Plotting
#plt.figure(figsize=(5, 5))
plt.plot(x_pursuer, y_pursuer, label='Pursuer curve (no prediction)', color='blue')
plt.scatter(x_pursuer[0], y_pursuer[0], color='blue')
plt.plot(x_pursuer_predicted, y_pursuer_predicted, label='Pursuer curve (with prediction)', color='blue', linestyle='--')
plt.plot(x_target_predicted, y_target_predicted, label='Target curve', color='red')
plt.scatter(x_target[0], y_target[0], color='red')

if len(solution.t_events[0]) > 0:
    plt.scatter(x_pursuer[-1], y_pursuer[-1], marker='x', color='green', label='Intersection')
else:
    plt.scatter(x_pursuer[-1], y_pursuer[-1], marker='x', color='blue', label='Pursuer')
    plt.scatter(x_target[-1], y_target[-1], marker='x', color='red', label='Target')

if len(solution_with_prediction.t_events[0]) > 0:
    plt.scatter(x_pursuer_predicted[-1], y_pursuer_predicted[-1], marker='D', color='green', label='Intersection (with prediction)')
else:
    plt.scatter(x_pursuer_predicted[-1], y_pursuer_predicted[-1], marker='D', color='blue', label='Pursuer (with prediction)')

plt.title('Pursuit Curve with Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()