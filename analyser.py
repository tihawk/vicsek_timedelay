#!/usr/bin/python
import numpy as np
import os
from correlation import static_correlation, spattemp_correlation,\
     time_zero, criticality_x, clearMem, assign_dimvelflucts
from geometry3d import get_all_distances
import time

# helper function to calculate the Static Correlation Function
def calculate_stat_corr(N, L, dt, startingTime, maxT, kValues, folder):
    fName = "N{0}L{1:.1f}dt{2}T{3}".format(N, L, dt, maxT)
    fDir = os.path.join('particles', folder)
    
    for file in os.listdir(fDir):
        if file.startswith(fName):
            positions = np.genfromtxt(os.path.join(fDir, file), skip_header=startingTime*(N+1))
            print(len(positions))
    
    time = 0
    absTime = startingTime
    statCorr = np.zeros(kValues.shape)
    count = 0
    nearest = 0
    
    print('Working on N{}, L{}'.format(N, L))
    while absTime < maxT-1:
        
        prevPos = positions[ time*(N) : time*(N)+N ]
        currentPos = positions[ time*(N)+N : time*(N)+N+N ]
        
        distances = get_all_distances(prevPos, L)
        nearest += criticality_x(distances)

        time = time + 1
        absTime += 1
        
        if(len(currentPos)<N):
            break
        
        statCorr += static_correlation(prevPos, currentPos, kValues, L )
        count += 1
        
    nearest /= count
    statCorr = statCorr / count 
    return statCorr, nearest

# helper function to calculate the Spatio-temporal Correlation Function
def calculate_spattemp_corr(N, L, dt, startingTime, maxT, timeLength, kValue, folder, grain=1):
    # clear the lookup tables in correlation.py
    clearMem()
    # file to write to
    fName = "N{0}L{1:.1f}dt{2}T{3}".format(N, L, dt, maxT)
    fDir = os.path.join('particles', folder)
    for file in os.listdir(fDir):
        if file.startswith(fName):
            # load position data from simulation file
            positions = np.genfromtxt(os.path.join(fDir, file), skip_header=startingTime*(N+1))
            print(len(positions))
            
    timeLength = timeLength // grain

    spattempCorr = np.zeros((timeLength, 2))
    
    timeFor = time.time()
        
    for index in range(timeLength):
        
        # time of correlation (relative to t0)
        timeI = 0
        # absolute time from beginning of simulation
        absTime = startingTime
        # time 0 of correlation (relative)
        t0 = 0
        # count of datapoint to average from (t_max-t)
        count = 0

        print( 'Reached t = {} / {} in {:.3f}s'.format( index*grain, grain*timeLength, (time.time()-timeFor) ) )
        timeFor = time.time()
        # while the absolute time is before end of simulation time
        while absTime < maxT-index*grain - 1:
            
            # move t0 from starting time of corr calculation towards tmax - t
            ind = t0*(N)
            tZeroPos = positions[ ind : ind+N ]
            # and load data for current t0
            tZeroPlusOnePos = positions[ ind+N*grain : ind+N+N*grain ] 
            time_zero(tZeroPos, tZeroPlusOnePos, t0, L)

            ind = (timeI+index*grain)*(N)
            prevPos = positions[ ind : ind+N ] 
            currentPos = positions[ ind+N*grain : ind+N+N*grain ] 
            
            if(len(currentPos)<N):
                break
            
            assign_dimvelflucts(tZeroPos, tZeroPlusOnePos, prevPos, currentPos, L, t0, (timeI+index*grain))
            
            spattempCorr[index, 1] += spattemp_correlation(prevPos, currentPos, kValue, L, (timeI+index*grain) ) 
            count += 1
            t0 += 1
            timeI += 1
            absTime += 1
            
        print(count)
        spattempCorr[index, 1] /= count
        spattempCorr[index, 0] = index*grain
                
    spattempCorr[:, 1] /= spattempCorr[0, 1]
    clearMem()
    return spattempCorr