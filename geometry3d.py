#!/usr/bin/python
import numpy as np 
#from scipy.spatial.distance import pdist, squareform, cdist
from math import pi, sin, cos, sqrt

from numba import jit

# generate random angle theta between -pi - pi
def rand_vector():
    theta = np.random.uniform(0,2*pi)
    z = np.random.uniform(-1,1)
    x = cos(theta) * sqrt(1 - z**2)
    y = sin(theta) * sqrt(1 - z**2)
    return np.array([x,y,z])
    
#@jit("float64(float64[:], float64[:])")
#def wrapped_euclidean_points(p, q):
#    diff = np.abs(p - q)
#    diff = diff - np.rint(diff / box_size) * box_size
#    return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
#
#@jit("float64[:](float64[:])")
# generate a lookup matrix table of distances between particles at time t
#def get_all_distances(particles):
#    distances = pdist(particles)
#    # NOTE: I don't know why, but squareform is faster than condensed
#    distances = squareform(distances)
##    distances = cdist(particles, particles, metric=wrapped_euclidean_points)
#    return distances

#@guvectorize(['float64[:,:], float64[:,:]'], '(m, n) -> (m, m)')
@jit
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
            res[i, j] = (dx**2 + dy**2 + dz**2)**0.5

            
    return res

"""NEIGHBOURS"""

# returns a list of indices for all neighbours
# includes itself as a neighor so it will be included in average
@jit()
def get_neighbours(distances, r, index):
    neighbours = []

    for j, dist in enumerate(distances[index]):
        if dist < r:
            neighbours.append(j)

    return neighbours

# average unit vectors for all angles
# return average angle 
@jit
def get_average(rand_vecs, neighbours):
	
    n_neighbours = len(neighbours)
    avg_vector = np.zeros(3)
    vec = rand_vecs[neighbours]
    avg_vector = np.sum(vec, axis=0)
    avg_vector = avg_vector / n_neighbours

    return avg_vector