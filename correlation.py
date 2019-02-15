import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import globals

#from numba import jit

# not doing this removes the benefits from numba???
globals.initialise()
box_size = globals.box_size

# list of values for the polarisation at times t (global)
polarisation = []
# list of variables for calculating the static correlation function at time
    # t_0 for normalising the spatio-temporal correlation (global vars)
particles_t0 = []
dimVelFluct3D_t0 = []
time0 = 0

# calculate the dimensionless velocity fluctuations used in corr calculations
#@jit
def dim_vel_fluctuations(particleVectors):

    global polarisation
    avgVector = np.mean(particleVectors, axis=0)
    polarisation.append(np.sqrt(np.dot(avgVector, avgVector)))
    velFluct = np.subtract(particleVectors, avgVector)
    sqAvgVelFluct = np.einsum('ij,ij->i', velFluct, velFluct)
    sqAvgVelFluct = np.mean(sqAvgVelFluct)
    dimVelFluct = velFluct / np.sqrt(sqAvgVelFluct)
        
    return dimVelFluct

# change coordinates to be wrt the centre of mass of the particles. Because
    # of the periodic boundary conditions, the coordinates are first mapped
    # onto a circle (see wiki)
#@jit
def coords_wrt_centre_mass(particles):
    theta = particles * 2*np.pi/box_size
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi = np.mean(xi, axis=0)
    zeta = np.mean(zeta, axis=0)
    theta = np.arctan2(-zeta, -xi) + np.pi
    cM = box_size*theta/(2*np.pi)

    return particles - cM

# get a matrix lookup table of all distances between particles
    # Static Correlation
#@jit
def distance_touple_stat(particles):
    particles = coords_wrt_centre_mass(particles)
    return squareform(pdist(particles))

# get a matrix lookup table of all distances between particles
    # Spatio-temporal Correlation
#@jit
def distance_touple(particles1, particles2):
    particles1 = coords_wrt_centre_mass(particles1)
    particles2 = coords_wrt_centre_mass(particles2)
    return cdist(particles1, particles2)

# static correlation function for a range of wavenumbers at time t
#@jit
def static_correlation(vectors, particles, kVal):    
    N = len(particles)
    deltaV = dim_vel_fluctuations(vectors)
    r_ij = distance_touple_stat(particles)
    
    statCorr = np.zeros(len(kVal))
    dVTensor = np.array([deltaV]*N)
    dottedDV = np.einsum('i...j,ij->i...', dVTensor, deltaV)
    
    for ind, wn in enumerate(kVal):
        kr = wn*r_ij
        temp = np.sin(kr)*dottedDV/kr
        temp = temp[~np.isnan(temp)]
        statCorr[ind] = np.sum(temp) / N
            
    return statCorr

# get the positions and unit vectors of the particles at t_0 for the purpose
    #of calculating the static correlation with wich to normalise the
    #spatio-temporal correlation
#@jit
def time_zero(particles, vectors, t):
    global particles_t0, dimVelFluct3D_t0, time0
    
    time0 = t
    particles_t0 = particles
    dimVelFluct_t0 = dim_vel_fluctuations(vectors)
    dimVelFluct3D_t0 = np.array([dimVelFluct_t0]*len(particles))
    
# spatio-temporal correlation function for wavenumber k at time t (not normalised)
#@jit
def spattemp_correlation(vectors, particles, k):    
    N = len(particles)
    deltaV = dim_vel_fluctuations(vectors)
    r_ij = distance_touple(particles_t0, particles)
    
    Corr = 0
    dottedDV = np.einsum('i...j,ij->i...', dimVelFluct3D_t0, deltaV)
    
    kr = k*r_ij
    temp = np.sin(kr)*dottedDV/kr
    temp = temp[~np.isnan(temp)]
    Corr = np.sum(temp) / N
            
    return Corr

""" ADDITIONAL """

# get susceptibility defined as the maximum of the static correlation function
def susceptibility(statCorr):
    return np.max(statCorr)
    
# get nearest neighbour distances at time t
def criticality_x(distances, r):
    nearest = np.zeros(len(distances[0]))
    
    for i in range(len(distances[0])):
        nearest[i] = np.min(distances[i][np.nonzero(distances[i])])
    
    r1 = np.mean(nearest)
    return r1/r