#!/usr/bin/env python

import numpy as np

# scikit-learn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances


# Main program

path = '/home/ubuntu/Desktop/Large_Scale_Tech/sparse_data_norm.npz'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/sparse_data_norm.npz'

loader = np.load(path)
sparse_data_norm = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
path = '/home/ubuntu/Desktop/Large_Scale_Tech/labels_arr.npy'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/labels_arr.npy'

labels_arr = np.load(path)

