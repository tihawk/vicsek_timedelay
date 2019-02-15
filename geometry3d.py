#!/usr/bin/python
import numpy as np 
#from scipy.spatial.distance import pdist, squareform, cdist
from math import pi, sin, cos, sqrt
import globals

from numba import jit, guvectorize, cuda

# not doing this removes the benefits from numba???
globals.initialise()
box_size = globals.box_size

# generate random angle theta between -pi - pi
def rand_vector():
    theta = np.random.uniform(0,2*pi)
    z = np.random.uniform(-1,1)
    x = cos(theta) * sqrt(1 - z**2)
    y = sin(theta) * sqrt(1 - z**2)
    return np.array([x,y,z])

# NOTE: linalg.norm is slower than sqrt for such small data quantity
#def wrapped_euclidean_points(p, q):
#    diff = np.abs(p - q)
#    return np.linalg.norm(np.minimum(diff, box_size - diff))
    
#@jit("float64(float64[:], float64[:])")
#def wrapped_euclidean_points(p, q):
#    diff = np.abs(p - q)
#    diff = diff - np.rint(diff / box_size) * box_size
#    return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
#
#@jit("float64[:](float64[:])")
## generate a lookup matrix table of distances between particles at time t
#def get_all_distances(particles):
#    distances = pdist(particles, metric=wrapped_euclidean_points)
#    # NOTE: I don't know why, but squareform is faster than condensed
#    distances = squareform(distances)
##    distances = cdist(particles, particles, metric=wrapped_euclidean_points)
#    return distances

@guvectorize(['float64[:,:], float64[:,:]'], '(m, n) -> (m, m)')
#@cuda.jit
def get_all_distances(ps, res):
    
    m = ps.shape[0]
#    n = ps.shape[1]
#    i, j = cuda.grid(2)
#    print(m)
#    print(n)
#    res = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
#    if i < m and j < m:
            dx = abs( ps[i,0] - ps[j,0] )
            dy = abs( ps[i,1] - ps[j,1] )
            dz = abs( ps[i,2] - ps[j,2] )
            dx = dx - np.rint(dx/box_size) * box_size
            dy = dy - np.rint(dy/box_size) * box_size
            dz = dz - np.rint(dz/box_size) * box_size
            res[i, j] = (dx**2 + dy**2 + dz**2)**0.5
            
#    return res
            
#def gpu_get_all_distances(particles):
#
#    rows = particles.shape[0]
#
#    block_dim = (16, 16)
#    grid_dim = (int(rows/block_dim[0] + 1), int(rows/block_dim[1] + 1))
#
#    stream = cuda.stream()
#    particles2 = cuda.to_device(np.asarray(particles, dtype=np.float64), stream=stream)
#    res2 = cuda.device_array((rows, rows))
#    get_all_distances[grid_dim, block_dim](particles2, res2)
#    out = res2.copy_to_host(stream=stream)
#
#    return out

# SLOWER FOR WHATEVER REASON...
# convert ij index to a condensed index for looking up
    # the distances condensed matrix
#@vectorize(["int32(int32, int32, int32)"])
#def square_to_condensed(i, j, n):
#    assert i != j, "no diagonal elements in condensed matrix"
#    if i < j:
#        i, j = j, i
#    return int(n*j - j*(j+1)/2 + i - 1 - j)

"""NEIGHBOURS"""

# returns a list of indices for all neighbours
# includes itself as a neighor so it will be included in average
@jit(nopython=True)
def get_neighbours(distances, r, index):
    neighbours = []

    for j, dist in enumerate(distances[index]):
        if dist < r:
            neighbours.append(j)

    return neighbours

# SLOWER FOR WHATEVER REASON
#@jit
#def get_neighbours(distances, r, index, N):
#    neighbours = []
#    indices = np.arange(N)
#    indices = np.delete(indices, index)
#    
#    dist_indices = np.array([square_to_condensed(indices, index, N), indices])
#    dist_indices = np.swapaxes(dist_indices, 0, 1)
##    print(dist_indices)
#    
##    dist_indices = []
##    for i, ind in enumerate(indices):
##        dist_indices.append([int(square_to_condensed(ind, index, N)), ind])
#        
#    for j, dind in enumerate(dist_indices):
#        if distances[dind[0]] < r:
#            neighbours.append(dind[1])
#    return neighbours


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