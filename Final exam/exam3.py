import matplotlib.pyplot as plt

## ODE, intervals, step size
def function(t,y): return 2*t - y
i1, i2, y0 = 0, 1, 10

## Euler's method
def euler(k, h = 0.001, y0 = y0, i1 = i1, i2 = i2):
    t, y, n = [i1], [y0], int((i2-i1)/h)
    for i in range(0, n):
        t.append(t[i]+h)
        y.append(y[i]+h*function(t[i],y[i]))
    plt.plot(t,y, color = 'red')
    return y[-1]
euler(5)
plt.ylabel("y")
plt.xlabel("t")
plt.show() ## Plot