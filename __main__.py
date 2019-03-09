#!/usr/bin/python
import sys
import numpy as np
from collections import deque
from geometry3d import rand_vector, get_all_distances, noise_application
from correlation import static_correlation, spattemp_correlation,\
     time_zero, polarisation, susceptibility, criticality_x
import time
import matplotlib.pyplot as plt

"""INITIALISE"""

"""Simulation Variables"""
# Set these before running!!!
# number of particles
N = int(sys.argv[1])

# size of system
box_size = float(sys.argv[2])

# length of time delay
timeDelay = int(sys.argv[3])

# are we running for static correlation (true) or spattemp corr (false)
isStatic = int(sys.argv[4])

# wavenumber for calculating the spattemp correlation
corrCalcK = float(sys.argv[5])

# noise intensity
eta = 0.45

# neighbour radius
r = 1.

# time step
t = 0
delta_t = 1

# the length of the dataset for the static correlation (in units of time)
staticTimeLength = 1000

# the length of the dataset for the spattemp correlation (in units of time)
timeLength = 200

# maximum time steps
if isStatic==1:
    T = 20000*delta_t
elif isStatic==0:
    if timeDelay == 0:
        T = 100000*delta_t
        # the length of the dataset for the spattemp correlation (in units of time)
        timeLength = 200
    elif timeDelay > 0:
        T = 200000*delta_t
        # the length of the dataset for the spattemp correlation (in units of time)
        timeLength = 500

# velocity of particles
vel = 0.05

# the time at which to start the spattemp corr calculations (in ratio of T)
corrCalcStart = 0.1*T
"""END Sim Vars"""

# make noise equilibration
noiseWidth = eta*np.pi

# initialise random particle positions
particles = np.random.uniform(0,box_size,size=(N,3))

# initialise random unit vectors in 3D
rand_vecs = np.zeros((N,3))
for i in range(0,N):
    vec = rand_vector()
    rand_vecs[i,:] = vec
    
noiseVecs = np.zeros((N, 3))
    
# init static correlation time average
if N < 128:
    wavenums = np.linspace(0., 2.0, num=45)
else:
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

def timestep(particles, rand_vecs):
    
    # actual simulation timestep
    for i in range(len(particles)):
        # get neighbor indices for current particle
#        neighbours = get_neighbours(distances, r, i)
        neighbours = np.where(distances[i]<r)
    
        # get average direction vector of neighbours
#        avg = get_average(rand_vecs, neighbours[0])
        avg = np.mean(rand_vecs[neighbours[0]], axis=0)
    
        # get noise vector from a uniform random distribution inside solid angle 4pi*eta
        # apply the noise vector by rotating it to the axis of the particle vector
        new_dir = noise_application(noiseWidth, avg)
        
        #NOTE: ^^^ PLEASE FOR CRYING OUT LOUD MAKE THIS FASTER!!!1 ^^^
        
        # add new unit vector to queue for current particle
        updtQueue[i].append(new_dir)
        
        # if the queue is long enough, dequeue and change unit vector accordingly
        # otherwise continue on previous trajectory
        if(len(updtQueue[i]) > timeDelay):
            rand_vecs[i] = updtQueue[i].popleft()
            # move to new position 
            particles[i,:] = particles[i,:] + delta_t * vel * rand_vecs[i]
        else:
            # move to new position using old unit vector
            particles[i,:] = particles[i,:] + delta_t * vel * rand_vecs[i]
        
    # assure correct boundaries (xmax,ymax) = (box_size, box_size)
    particles = np.where(particles < 0, particles + box_size, particles)
    particles = np.where(particles > box_size, particles - box_size, particles)



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
    distances = get_all_distances(particles, box_size)
    
    """StatCorr"""
    # in the last 10% of the simulation time, start calculating the static
    # correlation function at time t for a range of wavenumbers, in order to
    # build up an average. Meanwhile get the nearest neighbour distance at
    # time t. Later that will be averaged out as well.
    if(isStatic==1):
        if t >= 9999:#>= (T - 0.5*T - 1):
            if (counter == staticTimeLength):
                # if a length of the dataset is acquired, here the results are
                # averaged and outputted in a file. the new dataset starts at
                # the same timestep
                print("Starting new Dataset")
                
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
                
                statCorrTimeAvg = 0
                critX = 0
                counter = 0
                
            start = time.time()
        
            data = static_correlation(rand_vecs, particles, wavenums, box_size)
            
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
            
        if t >= corrCalcStart:
            # this construction will build up a few data sets of a set
                # time length, which will be later averaged out           
            if( corrIndex[0] < len(spatTempCorr[0]) ):
                # get time zero vars for spattempcorr
                if( corrIndex[1] == 0 ):
                    time_zero(particles, rand_vecs, t)
                    statCorrNormalisation = spattemp_correlation(
                            rand_vecs, particles, corrCalcK, box_size
                            )
                    
                if( corrIndex[1] < len(spatTempCorr) ):
                    spatTempCorr[corrIndex[1]][corrIndex[0]] =\
                        spattemp_correlation(
                                rand_vecs, particles, corrCalcK, box_size
                                ) / statCorrNormalisation
                    corrIndex[1] += 1
                else:
                    corrIndex[1] = 0
                    corrIndex[0] += 1
                    print("Starting new dataset number {}".format(corrIndex[0]))
            	
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
    # NOTE: Moved this up into the loop, so that more datasets can be saved
    # during simulation time
#    if(isStatic==1):
#        statCorrTimeAvg = statCorrTimeAvg / counter
#        
#        f=open("statCorr_{0}_{1}_{4}delay_{3}steps_{2}.txt".format(
#                N, box_size, time.time(), T, timeDelay),'ba'
#            )
#        output = np.concatenate((wavenums, statCorrTimeAvg),axis=0)
#        np.savetxt(f,output)
#        f.close()
#        
#        susc = susceptibility(statCorrTimeAvg)
#        critX = critX / counter
#        print("susc: {}. x: {}".format(susc, critX))
#        
#        f=open("{0}_{1}_{4}delay_{3}steps_{2}.txt".format(
#                N, box_size, time.time(), T, timeDelay),'ba'
#            )
#        output = np.array([susc, critX])
#        np.savetxt(f,output)
#        f.close()
        
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