#!/usr/bin/python
import numpy as np 
from scipy.spatial.distance import cdist
from math import pi, sin, cos, sqrt
from __main__ import box_size


# generate random angle theta between -pi - pi
def rand_vector():
    theta = np.random.uniform(0,2*pi)
    z = np.random.uniform(-1,1)
    x = cos(theta) * sqrt(1 - z**2)
    y = sin(theta) * sqrt(1 - z**2)
    return np.array([x,y,z])

# change coordinates to be wrt the centre of mass of the particles. Because
    # of the periodic boundary conditions, the coordinates are first mapped
    # onto a circle (see wiki)
def coords_wrt_centre_mass(particles):
    theta = particles * 2*np.pi/box_size
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi = np.mean(xi, axis=0)
    zeta = np.mean(zeta, axis=0)
    theta = np.arctan2(-zeta, -xi) + np.pi
    cM = box_size*theta/(2*np.pi)

    return particles - cM

# generate a lookup matrix table of distances between particles at time t
def get_all_distances(particles):
    return cdist(particles, particles)