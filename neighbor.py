#!/usr/bin/python
import numpy as np
from geometry3d import *


# returns a list of indices for all neighbors
# includes itself as a neighor so it will be included in average
def get_neighbors(distances, r, index):
    neighbors = []

    for j, dist in enumerate(distances[index]):
        if dist < r:
            neighbors.append(j)

    return neighbors


# average unit vectors for all angles
# return average angle 
def get_average(rand_vecs, neighbors):
	
	n_neighbors = len(neighbors)
	avg_vector = np.zeros(3)

	for index in neighbors:
		vec = rand_vecs[index]
#		v2 = np.array([0,0,0])
#		uv = unit_vector(vec, v2)
		avg_vector += vec

	avg_vector = avg_vector / n_neighbors

	return avg_vector




