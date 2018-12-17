#!/usr/bin/python
import numpy as np
from geometry3d import *
from neighbor import *
from correlation import *
from spattempcorr import *
from susceptibility import *
import sys
import time
import matplotlib.pyplot as plt

"""VARS IF RAN MANUALLY"""
# number of particles
N = 1024

# noise intensity
eta = 0.45

# neighbor radius
r = 1.

# time step
t = 0.
delta_t = 1

# size of system
box_size = 14.

# maximum time steps
T = 200000.*delta_t
vel = 0.05

corrCalcK = 0.397

"""VARS IF RAN FROM SCRIPT"""
## number of particles
#N = int(sys.argv[1])
#
## noise intensity
#eta = float(sys.argv[2])
#
## neighbor radius
#r = float(sys.argv[3])
#
## time step
#delta_t = 1
#
## size of system
#box_size = float(sys.argv[4])

# maximum time steps
#T = float(sys.argv[5])*delta_t

# Generate random particle coordinations
# particles[i,0] = x
# particles[i,1] = y
particles = np.random.uniform(0,box_size,size=(N,3))

# initialize random angles in 3D
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

#init spattempcorr
timeLength = 200
spatTempCorr = np.zeros(shape=(timeLength, T/(timeLength*2)))
corrIndex = [0, 0]
    
if __name__ == "__main__":
    
    # Run until time ends
    while t < T:
        if t%10 == 0:
            print ("step {}. time for 10 steps {}".format(t, time.time()-timestepTime))
            timestepTime = time.time()
            
        # get all relative distances between particles before looking for neighbours
        distances = get_all_distances(particles)
        
        """StatCorr"""
#        if t >= (T - 0.1*T):
#            start = time.time()
            
#            data = static_correlation(rand_vecs, particles, wavenums)
#            
#            statCorrTimeAvg += data
#            counter = counter + 1
#            critX += criticality_x(distances, r)
#            print("cereal calc time: {}".format(time.time()-start))
           
        """SpatTempCorr"""
        # get time zero vars for spattempcorr
        if t == (0.5*T - delta_t):
            print("Here we go")
            time_zero(particles, rand_vecs, t)
            statCorrNormalisation = static_correlation(rand_vecs, particles, [corrCalcK])
            
        if t >= 0.5*T:
            #start = time.time()
            
            if( corrIndex[0] < len(spatTempCorr[0]) ):
                if( corrIndex[1] < len(spatTempCorr) ):
                    spatTempCorr[corrIndex[1]][corrIndex[0]] = spattemp_correlation(rand_vecs, particles, corrCalcK) / statCorrNormalisation
                    corrIndex[1] += 1
                else:
                    corrIndex[1] = 0
                    corrIndex[0] += 1
                    print("Here we go")
                    time_zero(particles, rand_vecs, t)
                    statCorrNormalisation = static_correlation(rand_vecs, particles, [corrCalcK])
                	
        #np.savetxt("{0}.txt".format(t), output)#"simulation1/%.2f.txt" % t, output)
        ## save coordinates & angle vectors
        #output = np.concatenate((particles,rand_vecs),axis=1)
        
        for i, (x, y, z) in enumerate(particles):
            # get neighbor indices for current particle
            neighbours = get_neighbors(distances, r, i)
        
            # get average theta vector
            avg = get_average(rand_vecs, neighbours)
        
            # get noise vector
            noise = np.random.uniform(0, eta) * rand_vector()
            new_dir = ( (avg + noise) / np.sqrt(np.dot(avg + noise, avg + noise)) )
        
            # move to new position 
            particles[i,:] += delta_t * vel * new_dir
        
            # get new angle vector
            rand_vecs[i] = new_dir
        
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
    
        # new time step
        t += delta_t
        
    else:
        """HEREAFTER WE CARE ABOUT THE STATIC CORRELATION"""
#        statCorrTimeAvg = statCorrTimeAvg / counter
#        
#        f=open("statCorr_{0}_{1}_{3}steps_{2}.txt".format(len(particles), box_size, time.time(), T),'ba')
#        output = np.concatenate((wavenums, statCorrTimeAvg),axis=0)
#        np.savetxt(f,output)
#        f.close()
#        
#        susc = susceptibility(statCorrTimeAvg)
#        critX = critX / counter
#        print("susc: {}. x: {}".format(susc, critX))
#        
#        f=open("{0}_{1}_{3}steps_{2}.txt".format(len(particles), box_size, time.time(), T),'ba')
#        output = np.array([susc, critX])
#        np.savetxt(f,output)
#        f.close()
#        
#        plt.subplot(2, 1, 1)
#        plt.plot(wavenums, statCorrTimeAvg)
#        plt.subplot(2, 1, 2)
#        plt.plot(range(len(polarisation)), polarisation)
#        plt.show()
        """END"""
        
        """HEREAFTER WE CARE ABOUT THE SPATIO-TEMPORAL CORRELATION"""
        #plt.plot(range(len(spatTempCorr)), spatTempCorr)
        
        spatTempCorr = np.mean(spatTempCorr, axis=1)
        
        f=open("spatTempCorr_{0}_{1}_{4}_{3}steps_{2}.txt".format(len(particles), box_size, time.time(), T, corrCalcK),'ba')
        output = spatTempCorr
        np.savetxt(f,output)
        f.close()
        
        plt.plot(range(len(spatTempCorr)), spatTempCorr)
        plt.show()
        """END"""
        
        
        