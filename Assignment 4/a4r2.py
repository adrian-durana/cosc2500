import math
import numpy as np

import time
start_time = time.time()

from concurrent.futures import ProcessPoolExecutor

# n-sphere, centre at [0], length |1|

n_points = int(1e8)
batch_size = int(1e6)  # Process data in batches to save memory
n_batches = n_points // batch_size

## Analytical solution of volume
def volume(dim):
    return np.pi**(dim/2)/(math.gamma(1+(dim/2))*(2**dim))

## Estimation of volume, one batch
def estimate(dim, i=0):
    for _ in range(n_batches):
        x =  2*np.random.rand(batch_size, dim) - 1
        distance = np.sum(x**2, axis=1)
        i += np.sum(distance <= 1)
    return i / n_points

## Parallelisation
if __name__ == '__main__':
    # Parallelisation
    dimensions = list(range(2, 11))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(estimate, dimensions))
    # Display results
    for i, est_volume in enumerate(results, start=2):
        print(f"Dimension {i}")
        print(f"Estimated volume: {est_volume}")
        print(f"True volume: {volume(i)}")

    print("Process finished --- %s seconds ---" % (time.time() - start_time))

## Create statistical estimate of result

"""
dimension2
Dimension 2
0.78550917
0.7853981633974483
Dimension 3
0.52367769
0.5235987755982989
Dimension 4
0.30845055
0.30842513753404244
Dimension 5
0.16447359
0.16449340668482262
Dimension 6
0.08075341
0.08074551218828077
Dimension 7
0.03693678
0.03691223414321407
Dimension 8
0.01587144
0.0158543442438155
Dimension 9
0.0064533
0.006442400200661536
Dimension 10
0.0024926
0.00249039457019272
Process finished --- 765.1065514087677 seconds ---
"""

"""
Dimension 2
Estimate: 0.78546951
Volume: 0.7853981633974483
Dimension 3
Estimate: 0.52364689
Volume: 0.5235987755982989
Dimension 4
Estimate: 0.30843034
Volume: 0.30842513753404244
Dimension 5
Estimate: 0.16449496
Volume: 0.16449340668482262
Dimension 6
Estimate: 0.08072716
Volume: 0.08074551218828077
Dimension 7
Estimate: 0.03688401
Volume: 0.03691223414321407
Dimension 8
Estimate: 0.01584472
Volume: 0.0158543442438155
Dimension 9
Estimate: 0.00643057
Volume: 0.006442400200661536
Dimension 10
Estimate: 0.00248255
Volume: 0.00249039457019272

Dimension 2
Estimated volume:: 0.78542181
True volume: 0.7853981633974483
Dimension 3
Estimated volume:: 0.52364185
True volume: 0.5235987755982989
Dimension 4
Estimated volume:: 0.30843168
True volume: 0.30842513753404244
Dimension 5
Estimated volume:: 0.16450233
True volume: 0.16449340668482262
Dimension 6
Estimated volume:: 0.08076279
True volume: 0.08074551218828077
Dimension 7
Estimated volume:: 0.0369305
True volume: 0.03691223414321407
Dimension 8
Estimated volume:: 0.01584563
True volume: 0.0158543442438155
Dimension 9
Estimated volume:: 0.00644529
True volume: 0.006442400200661536
Dimension 10
Estimated volume:: 0.0024915
True volume: 0.00249039457019272
Process finished --- 309.54183197021484 seconds ---

"""

"""
Dimension 2
Estimated volume: 0.78538896
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52359105
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30841782
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.1644927
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.08070717
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.03689528
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.0158434
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.00643767
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00248841
True volume: 0.00249039457019272
Process finished --- 39.940508127212524 seconds ---

Dimension 2
Estimated volume: 0.78539406
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52362959
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30841247
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.16454377
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.08074186
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.03692484
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.01586847
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.00644895
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00248687
True volume: 0.00249039457019272
Process finished --- 43.71023964881897 seconds ---

Dimension 2
Estimated volume: 0.78538334
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52356486
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30838875
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.16453046
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.0807382
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.03691912
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.01584485
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.00644512
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00249296
True volume: 0.00249039457019272
Process finished --- 38.105855226516724 seconds ---

Dimension 2
Estimated volume: 0.78531162
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52365772
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30839435
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.16445819
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.08077351
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.0369003
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.01585321
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.00643426
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00249443
True volume: 0.00249039457019272
Process finished --- 37.905290842056274 seconds ---

Dimension 2
Estimated volume: 0.78540004
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52364368
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30842551
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.16444111
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.08076352
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.03690564
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.01587839
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.0064404
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00249374
True volume: 0.00249039457019272
Process finished --- 40.45209002494812 seconds ---

Dimension 2
Estimated volume: 0.785322
True volume: 0.7853981633974483
Dimension 3
Estimated volume: 0.52366537
True volume: 0.5235987755982988
Dimension 4
Estimated volume: 0.30852745
True volume: 0.30842513753404244
Dimension 5
Estimated volume: 0.16447743
True volume: 0.16449340668482265
Dimension 6
Estimated volume: 0.08078358
True volume: 0.08074551218828077
Dimension 7
Estimated volume: 0.03693596
True volume: 0.036912234143214075
Dimension 8
Estimated volume: 0.01587133
True volume: 0.0158543442438155
Dimension 9
Estimated volume: 0.00644692
True volume: 0.006442400200661536
Dimension 10
Estimated volume: 0.00247979
True volume: 0.00249039457019272
Process finished --- 38.82205605506897 seconds ---
"""