import numpy as np
from math import sqrt, pi, sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate random angle in 3D

theta = np.random.uniform(0,2*pi)
z = np.random.uniform(-0.05,0.05)

x = cos(theta) * sqrt(0.05 - z**2)
y = sin(theta) * sqrt(0.05 - z**2)

print (x, y, z)

# plot point
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z,color="r")
ax.plot([0,x],[0,y],[0,z],color="r")
ax.set_xlim3d(-0.5, 0.5)
ax.set_ylim3d(-0.5, 0.5)
ax.set_zlim3d(-0.5, 0.5)
plt.show()