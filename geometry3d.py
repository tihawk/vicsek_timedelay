#!/usr/bin/python
from __future__ import division
import numpy as np
from math import pi, sin, cos, sqrt
from numba import jit
import quaternion as quat

# generate a noise vector inside a cone of angle nu*pi around the north pole
# [1] https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region
# rotate the generated noise vector to the axis of the particle vector
# [2] https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
def noise_application(noiseWidth, vector):
    
    # Generate a random vector in solid angle 4*pi*nu around north pole
    z = np.random.uniform(0., 1.) * (1 - cos(noiseWidth)) + cos(noiseWidth)
    phi = np.random.uniform(0., 1.) * 2 * np.pi
    x = sqrt(1 - z**2) * cos( phi )
    y = sqrt(1 - z**2) * sin( phi )
    
    # Rotate the noise vector to be in a cone around the directional vector
    # rotation axis
    # pole = np.array([0, 0, 1])
    vector = vector/ sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    u =  np.cross([0, 0, 1], vector)
    #u = u/norm(u)
    # rotation angle
    rotTheta = np.arccos(np.dot(vector, [0, 0, 1]))
    #prepare rot angle for quaternion
    axisAngle = 0.5*rotTheta * u / sqrt(u[0]**2 + u[1]**2 + u[2]**2)
    # rotation matrix
    #M = expm( np.cross( np.eye(3), u * rotTheta ) )
    # Quaternion stuff - pretty fast, compared to other stuff...
    vec = quat.quaternion(x, y, z)
    qlog = quat.quaternion(*axisAngle)
    q = np.exp(qlog)
    
    vPrime = q * vec * np.conjugate(q)
    
    return vPrime.imag
    

# generate random angle theta between -pi - pi
def rand_vector():
    theta = np.random.uniform(0,2*pi)
    z = np.random.uniform(-1,1)
    x = cos(theta) * sqrt(1 - z**2)
    y = sin(theta) * sqrt(1 - z**2)
    return np.array([x,y,z])

#@guvectorize(['float64[:,:], float64[:,:]'], '(m, n) -> (m, m)')
# [3] https://en.wikipedia.org/wiki/Periodic_boundary_conditions#(A)_Restrict_particle_coordinates_to_the_simulation_box
@jit(nopython=True)
def get_all_distances(ps, box_size):
    m = ps.shape[0]
    res = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            dx = abs( ps[i,0] - ps[j,0] )
            dy = abs( ps[i,1] - ps[j,1] )
            dz = abs( ps[i,2] - ps[j,2] )
            dx = dx - np.rint(dx/box_size) * box_size
            dy = dy - np.rint(dy/box_size) * box_size
            dz = dz - np.rint(dz/box_size) * box_size
            res[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
    return res

"""NEIGHBOURS"""

# returns a list of indices for all neighbours
# includes itself as a neighor so it will be included in average
#@jit(nopython=True)
#def get_neighbours(distances, r, index):
#    neighbours = []
#
#    for j, dist in enumerate(distances[index]):
#        if dist < r:
#            neighbours.append(j)
#
#    return neighbours
#
## average unit vectors for all angles
## return average angle 
#@jit
#def get_average(rand_vecs, neighbours):
#    
#    vec = rand_vecs[neighbours]
#    avg_vector = np.mean(vec, axis=0)
#    print(avg_vector)
#
#    return avg_vector