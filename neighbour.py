#!/usr/bin/python
import numpy as np


# returns a list of indices for all neighbours
# includes itself as a neighor so it will be included in average
def get_neighbours(distances, r, index):
    neighbours = []

    for j, dist in enumerate(distances[index]):
        if dist < r:
            neighbours.append(j)

    return neighbours


# average unit vectors for all angles
# return average angle 
def get_average(rand_vecs, neighbours):
	
	n_neighbours = len(neighbours)
	avg_vector = np.zeros(3)

	for index in neighbours:
		vec = rand_vecs[index]
		avg_vector += vec

	avg_vector = avg_vector / n_neighbours

	return avg_vector




