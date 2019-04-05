#!/usr/bin/python
import sys
import numpy as np
from collections import deque
from geometry3d import rand_vector, get_all_distances, noise_application
import time

"""INITIALISE"""

"""Simulation Variables"""
# Set these before running!!!
# number of particles
N = int(sys.argv[1])

# size of system
box_size = float(sys.argv[2])

# length of time delay
timeDelay = int(sys.argv[3])

# noise intensity
eta = 0.45

# neighbour radius
r = 1.

# time step
t = 0
delta_t = 1

# maximum time steps
T = 20000*delta_t

# velocity of particles
vel = 0.05

"""END Sim Vars"""

# make noise equilibration
noiseWidth = eta*np.pi

# initialise random particle positions
particles = np.random.uniform(0,box_size,size=(N,3))
updatePos = particles
prevPos = np.zeros(particles.shape)

# initialise random unit vectors in 3D
rand_vecs = np.zeros((N,3))
for i in range(0,N):
    vec = rand_vector()
    rand_vecs[i,:] = vec
    
noiseVecs = np.zeros((N, 3))
updateVecs = rand_vecs

timestepTime = time.time()

# init time delay
updtQueue = np.zeros((N), dtype=deque)
for i in range(N):
    updtQueue[i] = deque()
"""END INIT"""

def timestep(particles, rand_vecs):
    
    # actual simulation timestep
    for i in range(len(particles)):
        
        # get neighbor indices for current particle
        neighbours = np.where(distances[i]<r)
        neighbours = neighbours[0][ np.where( neighbours[0] != i ) ]
        
        neighsDirs = rand_vecs[neighbours]
        
        # add neighbours' directions to queue to be used after time delay interval
        updtQueue[i].append(neighsDirs)
        
        # if the queue is long enough, dequeue and change unit vector accordingly
        # otherwise continue on previous trajectory
        if(len(updtQueue[i]) > timeDelay):
            
            # get neighbours' directions from before time delay interval
            neighsDirs = updtQueue[i].popleft()
            
            # get average direction vector of neighbours
            avg = np.mean([rand_vecs[i], *neighsDirs], axis=0)
            
            # apply the noise vector by rotating it to the axis of the particle vector
            newVec = noise_application(noiseWidth, avg)
            
            # move to new position 
            updatePos[i,:] = updatePos[i,:] + delta_t * vel * newVec
            
            # get new unit vector vector
            updateVecs[i] = newVec
        else:
            # move to new position using old unit vector
            updatePos[i,:] = updatePos[i,:] + delta_t * vel * rand_vecs[i]
    
        # assure correct boundaries (xmax,ymax) = (box_size, box_size)
        if updatePos[i,0] < 0:
            updatePos[i,0] = box_size + updatePos[i,0]
    
        if updatePos[i,0] > box_size:
            updatePos[i,0] = updatePos[i,0] - box_size
    
        if updatePos[i,1] < 0:
            updatePos[i,1] = box_size + updatePos[i,1]
    
        if updatePos[i,1] > box_size:
            updatePos[i,1] = updatePos[i,1] - box_size
    
        if updatePos[i,2] < 0:
            updatePos[i,2] = box_size + updatePos[i,2]
    
        if updatePos[i,2] > box_size:
            updatePos[i,2] = updatePos[i,2] - box_size
            
    particles = updatePos
    rand_vecs = updateVecs
    
    return particles, rand_vecs

"""TIMESTEP AND THINGS TO DO WHEN VISITING"""    
# Run until time ends
timestr = time.strftime("%Y%m%d-%H%M%S")
f = open( 'N{0}L{1}dt{2}T{3}_{4}.txt'.format(N, box_size, timeDelay, T, timestr), 'a+' )
#fv = open( 'N{0}L{1}dt{2}T{3}_vectors.txt'.format(N, box_size, timeDelay, T), 'a+' )
while t < T:
    # print progress update and time spent on n steps
    if t%100 == 0:
        print ("step {} / {}: avg. time for 10 steps {:.3f}".format(
                t, T, (time.time()-timestepTime)/10
                ))
        timestepTime = time.time()
        
    """Timestep"""
    
    # get all relative distances between particles before looking for neighbours
    distances = get_all_distances(particles, box_size)
    
    particles, rand_vecs = timestep(particles, rand_vecs)
    
    # export data and advance to new time step
    np.savetxt(f, particles, header='timestep {0}'.format(t) )
#    np.savetxt(fv, rand_vecs, header='timestep {0}'.format(t) )
    t += delta_t
    
else:
    f.close()
#    fv.close()