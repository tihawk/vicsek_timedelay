from __future__ import division
import numpy as np
from math import sqrt
from numba import jit

# list of values for the polarisation at times t (global)
polarisation = []
# list of variables for calculating the static correlation function at time
    # t_0 for normalising the spatio-temporal correlation (global vars)
particles_t0 = []
#dimVelFluct3D_t0 = []
prevPos_t0 = []
time0 = 0

allDeltaVs_t0 = [[None]]*20000
allDeltaVs = [[None]]*200000

# calculate the dimensionless velocity fluctuations used in corr calculations
def dim_vel_fluctuations(particlesT, particlesT_1, box_size):
    
#    vecs = subtract_with_PB(particlesT, particlesT_1, box_size)
#    MV = np.mean(vecs, axis=0)
#    print(MV)
#    velFluct1 = np.subtract(vecs, MV)
#    sqAvgVelFluct1 = np.einsum('ij,ij->i', velFluct1, velFluct1)
#    sqAvgVelFluct1 = np.mean(sqAvgVelFluct1)
#    dimVelFluct1 = velFluct1 / np.sqrt(sqAvgVelFluct1)

    pT = unravel_pbc(particlesT_1, particlesT, box_size)
    rsmd, proT, matrix = procrustes(particlesT_1, pT, box_size, reflection=False)
    vecs = np.subtract(proT, particlesT_1)
    avgVec = np.mean(vecs, axis=0)
    velFluct = np.subtract(vecs, avgVec)
    sqAvgVelFluct = np.einsum('ij,ij->i', velFluct, velFluct)
    sqAvgVelFluct = np.mean(sqAvgVelFluct)
    dimVelFluct = velFluct / np.sqrt(sqAvgVelFluct)
    
    return dimVelFluct

def procrustes(X, Y, box_size, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

#    muX = X.mean(0)
#    muY = Y.mean(0)
#
#    X0 = X - muX
#    Y0 = Y - muY
    
    X0, muX = coords_wrt_centre_mass(X, box_size, bounded=False)
    Y0t, muYt = coords_wrt_centre_mass(Y, box_size, bounded=False)
    muY = unravel_pbc(np.asarray([muX]), np.asarray([muYt]), box_size)[0]
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

#    if reflection is not 'best':
#
#        # does the current solution use a reflection?
#        have_reflection = np.linalg.det(T) < 0
#
#        # if that's not what was specified, force another reflection
#        if reflection != have_reflection:
#            V[:,-1] *= -1
#            s[-1] *= -1
#            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

# brings particles which have crossed the boundary on their last step back to
# their previous position beyound the boundary. important for the Procrustes algo
@jit(nopython=True)
def unravel_pbc(ps1, ps2, box_size):
    m, n = ps1.shape
    res = ps2
    
    for i in range(m):
        for j in range(n):
            dx =  ps1[i,j] - ps2[i,j] 
            if (dx >   box_size * 0.5):
                res[i, j] += box_size
            if (dx <= -box_size * 0.5):
                res[i, j] -= box_size
    return res

# removes diagonal elements of an array, in case i=/=j is important
def remove_diagonal(A):
    
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    out = strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)
    
    return out

# correct subtraction of positions under periodic boundary conditions
@jit(nopython=True)
def subtract_with_PB(ps1, ps2, box_size):
    m, n = ps1.shape
    res = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            dx =  ps1[i,j] - ps2[i,j] 
            if (dx >   box_size * 0.5):
                dx = dx - box_size
            if (dx <= -box_size * 0.5):
                dx = dx + box_size
            res[i, j] = dx
    return res

# correct distance calculation under periodic BC. important, since the method
    # used to apply the centre of mass results in an asymmetric distribution
    # of the particles around the boundaries
@jit(nopython=True)
def get_all_distances(ps1, ps2, box_size):
    m = ps1.shape[0]
    res = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            dx = abs( ps1[i,0] - ps2[j,0] )
            dy = abs( ps1[i,1] - ps2[j,1] )
            dz = abs( ps1[i,2] - ps2[j,2] )
            dx = dx - np.rint(dx/box_size) * box_size
            dy = dy - np.rint(dy/box_size) * box_size
            dz = dz - np.rint(dz/box_size) * box_size
            res[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
    return res

# the technically correct way to apply the centre of mass under periodic BC.
    # not used, since it results in a pattern of the positions unrecognisable
    # to the Procrustes algorithm
@jit(nopython=True)
def apply_new_cm(ps, cm, box_size):
    m = ps.shape[0]
    
    for i in range(m):
        dx = ( ps[i, 0] - cm[0] )
        dy = ( ps[i, 1] - cm[1] )
        dz = ( ps[i, 2] - cm[2] )
        if (dx > box_size * 0.5):
            dx = dx - box_size
        if (dx <= - box_size * 0.5):
            dx = dx + box_size
        if (dy > box_size * 0.5):
            dy = dy - box_size
        if (dy <= - box_size * 0.5):
            dy = dy + box_size
        if (dz > box_size * 0.5):
            dz = dz - box_size
        if (dz <= - box_size * 0.5):
            dz = dz + box_size
        ps[i, 0] = dx
        ps[i, 1] = dy
        ps[i, 2] = dz
    return ps

# change coordinates to be wrt the centre of mass of the particles. Because
    # of the periodic boundary conditions, the coordinates are first mapped
    # onto a circle (see wiki)
def coords_wrt_centre_mass(particles, box_size, bounded=True):
    ps = particles.copy()
    theta = ps * 2*np.pi/box_size
    xi = np.cos(theta)
    zeta = np.sin(theta)
    xi = np.mean(xi, axis=0)
    zeta = np.mean(zeta, axis=0)
    theta = np.arctan2(-zeta, -xi) + np.pi
    cM = box_size*theta/(2*np.pi)   
    
    # apply cM simply by subtracting. this will result in an asymmetric
    # distribution of the particles, but that doesn't matter for distance
    # calculations, because of subtract_with_PB. It is important however for
    # the correct application of the Procrustes algorithm.
    #apply_new_cm(ps, cM, box_size)
    
    if bounded:
        return apply_new_cm(ps, cM, box_size)
    else:
        return ps-cM, cM
    

# get a matrix lookup table of all distances between particles
    # Static Correlation
#@jit
def distance_touple_stat(particles, box_size):
    particlesCM = coords_wrt_centre_mass(particles, box_size)
    return  get_all_distances(particlesCM, particlesCM, box_size) #squareform(pdist(particles))

# get a matrix lookup table of all distances between particles
    # Spatio-temporal Correlation
#@jit
def distance_touple(particles1, particles2, box_size):
    particles1CM = coords_wrt_centre_mass(particles1, box_size)
    particles2CM = coords_wrt_centre_mass(particles2, box_size)
    
    return get_all_distances(particles1CM, particles2CM, box_size) #cdist(particles1CM, particles2CM)

# static correlation function for a range of wavenumbers at time t
#@jit
def static_correlation(prevPos, particles, kVal, box_size):    
    N = len(particles)
    deltaV = dim_vel_fluctuations(particles, prevPos, box_size)
    r_ij = distance_touple_stat(prevPos, box_size)
    
#    statCorr = np.zeros(len(kVal))
    dVTensor = np.array([deltaV]*N)
    dottedDV = np.einsum('i...j,ij->i...', dVTensor, deltaV)
    
    r_ijTensor = np.array([r_ij]*len(kVal))
    krTensor = np.einsum('i...j,i->i...j', r_ijTensor, kVal)
    checkZero = np.where(krTensor>0, np.sin(krTensor)/krTensor, 1)
    temp = np.einsum('...ij,ij->...ij', checkZero, dottedDV)
    statCorr = np.sum(temp, axis=(1, 2)) / N
        
    return statCorr

# get the positions and unit vectors of the particles at t_0 for the purpose
    #of calculating the static correlation with which to normalise the
    #spatio-temporal correlation
#@jit
def time_zero(prevPos, particles, t, box_size):
    global particles_t0, prevPos_t0, time0
    time0 = t
    particles_t0 = particles.copy()
    prevPos_t0 = prevPos.copy()
    
#@jit
def spattemp_correlation(prevPos, particles, k, box_size, timeI): 
    global particles_t0, prevPos_t0, time0, allDeltaVs_t0, allDeltaVs
    
    N = len(particles)
    
    if allDeltaVs_t0[time0][0] is None:
        allDeltaVs_t0[time0] = dim_vel_fluctuations(particles_t0, prevPos_t0, box_size)
    if allDeltaVs[timeI][0] is None:
        allDeltaVs[timeI] = dim_vel_fluctuations(particles, prevPos, box_size)
    deltaV_t0 = allDeltaVs_t0[time0]#dim_vel_fluctuations(particles_t0, prevPos_t0, box_size)
    deltaV = allDeltaVs[timeI]#dim_vel_fluctuations(particles, prevPos, box_size)
    r_ij = distance_touple(prevPos_t0, prevPos, box_size)
    
    #r_ij = remove_diagonal(r_ij)
    
    dottedDV = np.einsum('i...j,ij->i...', np.array([deltaV_t0]*N), deltaV)
    
    #dottedDV = remove_diagonal(dottedDV)
    
    kr = k*r_ij
    # sin(x)/x -> 1 as x -> 0
    checkZero = np.where(kr>0, np.sin(kr)/kr, 1.)
    temp = checkZero*dottedDV
    Corr = np.sum(temp) / N
            
    return Corr

""" ADDITIONAL """

def clearMem():
    global particles_t0, prevPos_t0, time0, allDeltaVs_t0, allDeltaVs
    particles_t0 = []
    prevPos_t0 = []
    time0 = 0
    allDeltaVs_t0 = [[None]]*20000
    allDeltaVs = [[None]]*200000
    
def clearMemDeltaVs():
    global allDeltaVs
    
    allDeltaVs = [[None]]*200000

# get susceptibility defined as the maximum of the static correlation function
def susceptibility(statCorr):
    return np.max(statCorr)
    
# get nearest neighbour distances at time t
def criticality_x(distances, r=1.):
    nearest = np.zeros(len(distances[0]))
    
    for i in range(len(distances[0])):
        nearest[i] = np.min(distances[i][np.nonzero(distances[i])])
    
    r1 = np.mean(nearest)
    return r1/r