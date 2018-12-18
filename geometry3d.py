#!/usr/bin/python
import numpy as np 
from scipy.spatial.distance import cdist
from math import pi, sin, cos, sqrt


# generate random angle theta between -pi - pi
def rand_vector():
    theta = np.random.uniform(0,2*pi)
    z = np.random.uniform(-1,1)
    x = cos(theta) * sqrt(1 - z**2)
    y = sin(theta) * sqrt(1 - z**2)
    return np.array([x,y,z])

#def coords_wrt_centre_mass(particles):
#    cM = np.mean(particles, axis=0)
#    particles -= cM
#
#    return particles

# generate a lookup matrix table of distances between particles at time t
def get_all_distances(particles):
    return cdist(particles, particles)    