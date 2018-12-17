import numpy as np
from scipy.spatial.distance import cdist
from geometry3d import *
from neighbor import *
import sys
import matplotlib.pyplot as plt
import time

polarisation = []

def dim_vel_fluctuations(particleVectors):
#    N = len(particleVectors)
    
#    velFluct = np.zeros(shape=(N, 3))
#    dimVelFluct = np.zeros(shape=(N, 3))
#    avgVector = np.array([0., 0., 0.])
    
#    for vec in particleVectors:
#        avgVector += vec   
#        
#    avgVector /= N
        
    #delva v_i
#    for i, vec in enumerate(particleVectors):
#        velFluct[i] = vec - avgVector
    
    #sum of delta v_i^2  
#    print(velFluct)
#    sqAvgVelFluct = 0
#    
#    for vec in velFluct:
#        sqAvgVelFluct += np.dot(vec, vec)
#        
#    sqAvgVelFluct /= N
        
    #dimensionless velocity fluctuation
#    for i, vec in enumerate(velFluct):
#        dimVelFluct[i] = velFluct[i] / np.sqrt(sqAvgVelFluct)
    
    # vectorised
    global polarisation
    avgVector = np.mean(particleVectors, axis=0)
    polarisation.append(np.sqrt(np.dot(avgVector, avgVector)))
    velFluct = np.subtract(particleVectors, avgVector)
    sqAvgVelFluct = np.einsum('ij,ij->i', velFluct, velFluct)
    sqAvgVelFluct = np.mean(sqAvgVelFluct)
    dimVelFluct = velFluct / np.sqrt(sqAvgVelFluct)
        
    return dimVelFluct

def distance_touple(particles):
#    N = len(particles)
    #particles = coords_wrt_centre_mass(particles)
    
#    distTouple = np.zeros(shape=(N, N))
#    
#    for i, p1 in enumerate(particles):
#        for j, p2 in enumerate(particles):
##            distTouple[i, j] = np.linalg.norm(p1-p2)
#            distTouple[i, j] = euclidean_distance(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
    
    return cdist(particles, particles)

def static_correlation(vectors, particles, kVal):    
    N = len(particles)
    deltaV = dim_vel_fluctuations(vectors)
    r_ij = distance_touple(particles)
    
#    wavenum = np.linspace(0.1, 5, num=50)
    
    # vectorised
    statCorr = np.zeros(len(kVal))
    dVTensor = np.array([deltaV]*N)
    dottedDV = np.einsum('i...j,ij->i...', dVTensor, deltaV)
    
    for ind, wn in enumerate(kVal):
        kr = wn*r_ij
        temp = np.sin(kr)*dottedDV/kr
        temp = temp[~np.isnan(temp)]
        statCorr[ind] = np.sum(temp) / N
    
#    print(deltaV)
#    print(deltaV.shape)
    
#    dVTensor = np.zeros(shape=(N, N, 3))
#    for i in range(N):
#        dVTensor[i,:,:] = np.roll(deltaV, 0, axis=0)
        
        
#    print(dVTensor)
#    print(dVTensor.shape)
#    
#    print(np.einsum('i...j,ij->i...', dVTensor, deltaV))
#    print(np.dot(deltaV[0], deltaV[0]))
#    print(np.dot(deltaV[0], deltaV[1]))
#    print(np.dot(deltaV[0], deltaV[2]))
#    print(np.dot(deltaV[1], deltaV[0]))
#    print(np.dot(deltaV[1], deltaV[1]))
#    print(np.dot(deltaV[1], deltaV[2]))
#    print(np.dot(deltaV[2], deltaV[0]))
#    print(np.dot(deltaV[2], deltaV[1]))
#    print(np.dot(deltaV[2], deltaV[2]))
#    print(np.einsum('ij,ij->i', deltaV, np.roll(deltaV, 1, axis=0)))
#    print(np.einsum('ij,ij->ij', deltaV, deltaV))
    
   
#    print(dVTensor)
#    dVTensor = np.einsum('ijk,ijk->ik...', dVTensor, dVTensor)
#    print(dVTensor.shape)
    
    
    
#    for i, dV_i in enumerate(deltaV):
#        for j, dV_j in enumerate(deltaV):
##            for index, kVal in enumerate(wavenum):
##            print("{} vs {}".format(dVdotted[i, j], np.dot(dV_i, dV_j)))
#            temp = statCorr + ( ( np.sin(kr[i, j]) )* dottedDV[i, j] / ( kr[i, j] ) )
#            if ~np.isnan(temp):
#                statCorr = temp
#             
#    statCorr /= N
            
    return statCorr

#test
#testN = 512
#start = time.time()
#particles = np.random.uniform(0,1,size=(testN,3))
#rand_vecs = np.zeros((testN,3))
#for i in range(0,testN):
#	vec = rand_vector()
#	rand_vecs[i,:] = vec
#          
##dim_vel_fluctuations(rand_vecs)
#distance_touple(rand_vecs)
##data = static_correlation(particles, rand_vecs, range(50))
#
#print(time.time() - start)
##plt.plot(data[0], data[1])
##plt.show()