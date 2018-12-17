import numpy as np
from scipy.spatial.distance import cdist
from geometry3d import *
from neighbor import *
import sys
import matplotlib.pyplot as plt
import time

polarisation = []
particles_t0 = []
dimVelFluct3D_t0 = []
time0 = 0

def spattemp_dim_vel_fluctuations(particleVectors):
    
    # vectorised
    global polarisation
    avgVector = np.mean(particleVectors, axis=0)
    polarisation.append(np.sqrt(np.dot(avgVector, avgVector)))
    velFluct = np.subtract(particleVectors, avgVector)
    sqAvgVelFluct = np.einsum('ij,ij->i', velFluct, velFluct)
    sqAvgVelFluct = np.mean(sqAvgVelFluct)
    dimVelFluct = velFluct / np.sqrt(sqAvgVelFluct)
        
    return dimVelFluct

def time_zero(particles, vectors, t):
    global particles_t0, dimVelFluct3D_t0, time0
    
    time0 = t
    particles_t0 = particles#coords_wrt_centre_mass(particles)
    dimVelFluct_t0 = spattemp_dim_vel_fluctuations(vectors)
    dimVelFluct3D_t0 = np.array([dimVelFluct_t0]*len(particles))

def spattemp_distance_touple(particles_t):
#    N = len(particles)
    #particles_t = coords_wrt_centre_mass(particles_t)
    
    return cdist(particles_t0, particles_t)

def spattemp_correlation(vectors, particles, k):    
    N = len(particles)
    deltaV = spattemp_dim_vel_fluctuations(vectors)
    r_ij = spattemp_distance_touple(particles)
    
    # vectorised
    Corr = 0
#    dVTensor = np.array([deltaV]*N)
    dottedDV = np.einsum('i...j,ij->i...', dimVelFluct3D_t0, deltaV)
    
    kr = k*r_ij
    temp = np.sin(kr)*dottedDV/kr
    temp = temp[~np.isnan(temp)]
    Corr = np.sum(temp) / N
            
    return Corr