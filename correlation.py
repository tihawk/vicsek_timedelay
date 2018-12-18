import numpy as np
from scipy.spatial.distance import cdist

# list of values for the polarisation at times t (global)
polarisation = []
# list of variables for calculating the static correlation function at time
    # t_0 for normalising the spatio-temporal correlation (global vars)
particles_t0 = []
dimVelFluct3D_t0 = []
time0 = 0

# calculate the dimensionless velocity fluctuations used in corr calculations
def dim_vel_fluctuations(particleVectors):

    global polarisation
    avgVector = np.mean(particleVectors, axis=0)
    polarisation.append(np.sqrt(np.dot(avgVector, avgVector)))
    velFluct = np.subtract(particleVectors, avgVector)
    sqAvgVelFluct = np.einsum('ij,ij->i', velFluct, velFluct)
    sqAvgVelFluct = np.mean(sqAvgVelFluct)
    dimVelFluct = velFluct / np.sqrt(sqAvgVelFluct)
        
    return dimVelFluct

# get a matrix lookup table of all distances between particles
def distance_touple(particles1, particles2):
    return cdist(particles1, particles2)

# static correlation function for a range of wavenumbers at time t
def static_correlation(vectors, particles, kVal):    
    N = len(particles)
    deltaV = dim_vel_fluctuations(vectors)
    r_ij = distance_touple(particles, particles)
    
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
def time_zero(particles, vectors, t):
    global particles_t0, dimVelFluct3D_t0, time0
    
    time0 = t
    particles_t0 = particles#coords_wrt_centre_mass(particles)
    dimVelFluct_t0 = dim_vel_fluctuations(vectors)
    dimVelFluct3D_t0 = np.array([dimVelFluct_t0]*len(particles))
    
# spatio-temporal correlation function for wavenumber k at time t (not normalised)
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