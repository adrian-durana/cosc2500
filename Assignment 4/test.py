import numpy as np
import scipy as sp

def estimate(dim, points):
    # Use only the first 'dim' coordinates of the points
    distances = np.sum(points[:, :dim]**2, axis=1)  # Squared distances for the current dimension
    inside_sphere = np.sum(distances <= 1)  # Count points inside the unit hypersphere
    ratio = inside_sphere / len(points)
    return ratio

def volume(dim):
    # Example placeholder function; replace with actual implementation for volume if needed.
    return np.pi**(dim/2)/(sp.special.gamma(1 + (dim/2))*(2**dim))

# Generate all random points once, covering the highest dimension needed
n_points = int(1e8)
max_dim = 10  # Set the highest dimension for which we need to estimate
points = 2 * np.random.rand(n_points, max_dim) - 1  # Generate random points in [-1, 1] for the highest dimension

# Results for dimensions 2 to 10
for i in range(2, 11):
    print(f"Dimension {i}")
    print(f"Estimate: {estimate(i, points)}")
    print(f"Volume: {volume(i)}")

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
"""