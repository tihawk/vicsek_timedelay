import numpy as np
from geometry3d import *
from neighbor import *
from __main__ import box_size, eta, particles, delta_t, rand_vecs, r

def timestep(i, x, y, z):
    # get neighbor indices for current particle
    neighbours = get_neighbors(particles, r, x, y, z)

    # get average theta vector
    avg = get_average(rand_vecs, neighbours)

    # get noise vector
    noise = eta * rand_vector()

    # move to new position 
    particles[i,:] += delta_t * (avg + noise)

    # get new angle vector
    rand_vecs[i] = avg + noise

    # assure correct boundaries (xmax,ymax) = (1,1)
    if particles[i,0] < 0:
        particles[i,0] = box_size + particles[i,0]

    if particles[i,0] > box_size:
        particles[i,0] = particles[i,0] - box_size

    if particles[i,1] < 0:
        particles[i,1] = box_size + particles[i,1]

    if particles[i,1] > box_size:
        particles[i,1] = particles[i,1] - box_size

    if particles[i,2] < 0:
        particles[i,2] = box_size + particles[i,2]

    if particles[i,2] > box_size:
        particles[i,2] = particles[i,2] - box_size
