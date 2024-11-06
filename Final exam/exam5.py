import numpy as np

def afactors (a , b , c ):
# Usage : [ A_a A_b A_c ] = afactors ( a , b , c )
# Bottcher vol 1 pg 80

    current_warning_state = np.seterr(all='ignore')
    ma = min(a,b,c)
    intervals = 20
    polyorder = 8
    tol = 1e-8

    alpha = np.array([ a , b , c ])
    polyint = 1.0 / np.arange ( polyorder + 1 , 0 , -1)

    afac = np.zeros (3)

    for n in range (3):
        total = 0
        startx = 0
        endx = ma
        err1 = 1
        while err1 > tol:
            x = startx + np.arange (intervals + 1) / intervals * (endx - startx)
            R = np.sqrt ((x + a**2) * (x + b**2) * (x + c**2))
            y = 1.0 / (x + alpha [ n ]**2) / R
            P = np.polyfit(x, y, polyorder)
            Pint = P * polyint
            Pint = np . append (Pint , 0) # Constant term , polyval () will need this
            yint = np . polyval (Pint, [startx , endx])
            contribhere = yint[1] - yint[0] # = diff ( yint ) * blah
            total += contribhere
            err1 = contribhere / total
            startx = endx
            endx = startx * 2
        afac[n] = total

    afac = afac * a * b * c / 2

    np.seterr(**current_warning_state)

    return afac