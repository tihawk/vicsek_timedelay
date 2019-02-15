#!/usr/bin/python
import sys
import numpy as np
from collections import deque
import globals
from geometry3d import rand_vector, get_all_distances\
     ,get_neighbours, get_average
from correlation import static_correlation, spattemp_correlation,\
     time_zero, polarisation, susceptibility, criticality_x
import time
import matplotlib.pyplot as plt

from numba import guvectorize

"""INITIALISE"""
globals.initialise()
globals.N = int(sys.argv[1])
globals.box_size = float(sys.argv[2])
globals.timeDelay = int(sys.argv[3])
globals.isStatic = int(sys.argv[4])
globals.corrCalcK = float(sys.argv[5])
if globals.isStatic==1:
    globals.T = 10000*globals.delta_t
    globals.corrCalcStart = 0.1*globals.T
elif globals.isStatic==0:
    globals.T = 100000*globals.delta_t
    globals.corrCalcStart = 0.1*globals.T
# number of particles
N = globals.N
# size of system
box_size = globals.box_size
# noise intensity
eta = globals.eta
# neighbour radius
r = globals.r
# time step
t = globals.t
delta_t = globals.delta_t
# maximum time steps
T = globals.T
# velocity of particles
vel = globals.vel
# length of time delay
timeDelay = globals.timeDelay
# are we running for static correlation (true) or spattemp corr (false)
isStatic = globals.isStatic
# wavenumber for calculating the spattemp correlation
corrCalcK = globals.corrCalcK
# the length of the dataset for the spattemp correlation (in units of time)
timeLength = globals.timeLength
# the time at which to start the spattemp corr calculations (in ratio of T)
corrCalcStart = globals.corrCalcStart

# initialise random particle positions
particles = np.random.uniform(0,box_size,size=(N,3))

# initialise random unit vectors in 3D
rand_vecs = np.zeros((N,3))
for i in range(0,N):
    vec = rand_vector()
    rand_vecs[i,:] = vec
    
# init static correlation time average
wavenums = np.linspace(0., 1.5, num=35)
statCorrTimeAvg = np.zeros(len(wavenums))
critX = 0
counter = 0
timestepTime = time.time()

# init spattempcorr
smtin = [np.int(timeLength), np.int( (T - corrCalcStart)/(timeLength) )]
spatTempCorr = np.zeros( shape=( smtin ) )
corrIndex = [0, 0]

# init time delay
updtQueue = np.zeros((N), dtype=deque)
for i in range(N):
    updtQueue[i] = deque()
"""END INIT"""

#@guvectorize(["void(float64[:,:], float64[:,:], float64[:,:], float64[:,:])"],\
#              '(m, n), (m, n) -> (m, n), (m, n)')
def timestep(particles, rand_vecs):
    # actual simulation timestep
    for i, (x, y, z) in enumerate(particles):
        # get neighbor indices for current particle
        neighbours = get_neighbours(distances, r, i)
    
        # get average theta vector
        avg = get_average(rand_vecs, neighbours)
    
        # get noise vector
        noise = eta*rand_vector()#np.random.uniform(0, eta) * rand_vector()
        
        # calculate the new unit vector for the particle, taking into account
        # the noise, and of course - normalisation
        avgAndNoise = avg + noise
        try:
            new_dir = ( (avgAndNoise) / np.linalg.norm(avgAndNoise) )
        except:
            print("AHA!!!")
        
        # add new unit vector to queue for current particle
        updtQueue[i].append(new_dir)
        
        # if the queue is long enough, dequeue and change unit vector accordingly
        # otherwise continue on previous trajectory
        if(len(updtQueue[i]) > timeDelay):
            newVec = updtQueue[i].popleft()
            # move to new position 
            particles[i,:] = particles[i,:] + delta_t * vel * newVec
            # get new unit vector vector
            rand_vecs[i] = newVec
        else:
            # move to new position using old unit vector
            particles[i,:] = particles[i,:] + delta_t * vel * rand_vecs[i]
    
        # assure correct boundaries (xmax,ymax) = (box_size, box_size)
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



"""TIMESTEP AND THINGS TO DO WHEN VISITING"""    
# Run until time ends
while t < T:
    # print progress update and time spent on n steps
    if t%100 == 0:
        print ("step {}: avg. time for 10 steps {:3f}".format(
                t, (time.time()-timestepTime)/10
                ))
        timestepTime = time.time()
        
    # get all relative distances between particles before looking for neighbours
    distances = get_all_distances(particles)
    
    """StatCorr"""
    # in the last 10% of the simulation time, start calculating the static
    # correlation function at time t for a range of wavenumbers, in order to
    # build up an average. Meanwhile get the nearest neighbour distance at
    # time t. Later that will be averaged out as well.
    if(isStatic==1):
        if t >= (T - 0.1*T):
            start = time.time()
        
            data = static_correlation(rand_vecs, particles, wavenums)
            
            statCorrTimeAvg += data
            counter = counter + 1
            critX += criticality_x(distances, r)
            print("cereal calc time: {:3f}".format(time.time()-start))
       
    """SpatTempCorr"""
    # after a certain amount of steps (i.e. 50%), start calculating the spatio-
    # temporal correlation function. That will be calculated for a certain
    # amount of steps, after which a new dataset will be created, and the process
    # repeated. Later those will be averaged out.
    if(isStatic==0):
        # get time zero vars for spattempcorr
        if t == (corrCalcStart - delta_t):
            print("Here we go")
            time_zero(particles, rand_vecs, t)
            statCorrNormalisation = spattemp_correlation(
                    rand_vecs, particles, corrCalcK
                    )
            
        if t >= corrCalcStart:
            # this construction will build up a few data sets of a set
                # time length, which will be later averaged out            
            if( corrIndex[0] < len(spatTempCorr[0]) ):
                if( corrIndex[1] < len(spatTempCorr) ):
                    spatTempCorr[corrIndex[1]][corrIndex[0]] =\
                        spattemp_correlation(
                                rand_vecs, particles, corrCalcK
                                ) / statCorrNormalisation
                    corrIndex[1] += 1
                else:
                    corrIndex[1] = 0
                    corrIndex[0] += 1
                    print("Starting new dataset number {}".format(corrIndex[0]))
                    time_zero(particles, rand_vecs, t)
                    statCorrNormalisation =\
                        spattemp_correlation(rand_vecs, particles, corrCalcK)
            	
    #np.savetxt("{0}.txt".format(t), output)#"simulation1/%.2f.txt" % t, output)
    ## save coordinates & angle vectors
    #output = np.concatenate((particles,rand_vecs),axis=1)
    """Timestep"""
    
    timestep(particles, rand_vecs)
    
    # new time step
    t += delta_t
    
else:
    """HEREAFTER WE CARE ABOUT THE STATIC CORRELATION"""
    # here the data from the static correlation will be averaged, saved and plotted.
    # the susceptibility (defined as the maximum of the static correlation) and
    # nearest neighbour distance, will be averaged, saved and printed as well.
    # the polarisation will be printed as well
    if(isStatic==1):
        statCorrTimeAvg = statCorrTimeAvg / counter
        
        f=open("statCorr_{0}_{1}_{4}delay_{3}steps_{2}.txt".format(
                N, box_size, time.time(), T, timeDelay),'ba'
            )
        output = np.concatenate((wavenums, statCorrTimeAvg),axis=0)
        np.savetxt(f,output)
        f.close()
        
        susc = susceptibility(statCorrTimeAvg)
        critX = critX / counter
        print("susc: {}. x: {}".format(susc, critX))
        
        f=open("{0}_{1}_{4}delay_{3}steps_{2}.txt".format(
                N, box_size, time.time(), T, timeDelay),'ba'
            )
        output = np.array([susc, critX])
        np.savetxt(f,output)
        f.close()
        
#        plt.plot(wavenums, statCorrTimeAvg)
#        plt.show()
#        plt.plot(range(len(polarisation)), polarisation)
    """END STATCORR"""
    
    """HEREAFTER WE CARE ABOUT THE SPATIO-TEMPORAL CORRELATION"""
    # here the data for the spatio-temporal correlation will be averaged,
        # saved and plotted.
    # the polarisation will be printed as well
    if(isStatic==0):
        #plt.plot(range(len(spatTempCorr)), spatTempCorr)
        
        print(spatTempCorr[:][:5])
        spatTempCorr = np.mean(spatTempCorr, axis=1)
        
        f=open("spatTempCorr_{0}_{1}_{5}delay_{4}_{3}steps_{2}.txt".format(
                N, box_size, time.time(), T, corrCalcK, timeDelay),'ba'
            )
        output = spatTempCorr
        np.savetxt(f,output)
        f.close()
        
#        plt.plot(range(len(spatTempCorr)), spatTempCorr)
#        plt.show()
#        plt.plot(range(len(polarisation)), polarisation)
    """END SPATTEMPCORR"""