import matplotlib.pyplot as plt
from convexhull import convex_hull
from scipy.interpolate import spline
import numpy as np

plt.xlabel("Number of points")
plt.ylabel("Time taken (seconds)")

times = [] # Y Axis
points = np.array([10, 25, 50, 75, 100, 500, 1000, 5000, 10000, 15000, 20000, 50000, 100000]) # X Axis

convex_hull(100000)

#plt.plot(points, times)

#plt.show()