import math
import numpy as np
from scipy import stats
import time
start_time = time.time()
from concurrent.futures import ProcessPoolExecutor

## Variables
n_points = int(1e8)
batch_size = int(1e6) # Process data in batches to save memory
n_batches = n_points // batch_size

## Analytical solution of volume
def volume(dim):
    return np.pi**(dim/2)/(math.gamma(1+(dim/2))*(2**dim))

## Estimation of volume, one batch
def estimate(dim, inside=0):
    for _ in range(n_batches):
        x =  2*np.random.rand(batch_size, dim) - 1
        distance = np.sum(x**2, axis=1)
        inside += np.sum(distance <= 1)
    return inside / n_points

if __name__ == '__main__':
    ## Parallelisation
    dimensions = list(range(2, 4))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(estimate, dimensions))
    ## Display results
    for i, est_volume in enumerate(results, start=2):
        print(f"{i}-ball")
        print(f"Estimated volume: {est_volume}")
        print(f"True volume: {volume(i)}")
        ## Create statistical estimate of result
        print(f"Error: {abs(volume(i) - est_volume)}")
        error = stats.norm.ppf(0.95)*np.sqrt(est_volume*(1-est_volume)/n_points)
        C_a = np.floor((est_volume - error) * 10**6) / 10**6
        C_b = np.ceil((est_volume + error) * 10**6) / 10**6
        print(f"90% confidence interval: ({C_a},{C_b})")
    print("--- Execution time: %s seconds ---" % (time.time() - start_time))