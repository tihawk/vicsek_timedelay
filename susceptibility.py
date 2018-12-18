import numpy as np

def susceptibility(statCorr):
    return np.max(statCorr)
    
def criticality_x(distances, r):
    nearest = np.zeros(len(distances[0]))
    
    for i in range(len(distances[0])):
        nearest[i] = np.min(distances[i][np.nonzero(distances[i])])
    
    r1 = np.mean(nearest)
    return r1/r
    
