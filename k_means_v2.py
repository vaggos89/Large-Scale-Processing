#!/usr/bin/env python

import numpy as np

# scikit-learn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, cosine_similarity
import csv
import store_data_format as sdf
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


def rand_representatives(data, num_repr):

    # Initialize representatives randomly

    # Create an array [1, num_of_articles]
    temp_array = np.array(range(data.shape[0]))

    # Randomly shuffle the array
    np.random.shuffle(temp_array)

    # Select the firs 5, random representatives
    return data[temp_array[0:num_repr]]


def seq_representatives(data, num_repr):

    # Initialize representatives using a sequential
    return data


def longest_dist_representatives(data, num_per):

    # Initialize representatives using a sequential
    return data


def calc_dist(flag, data, curr_repr):

    # Calculate distances between the data and the representatives

    if flag == 0:
        return euclidean_distances(data, curr_repr)

    # tmp = cosine_similarity(data, curr_repr, dense_output=False).toarray()
    return cosine_distances(data, curr_repr)


def update_clusters(dist):

    # Update the clusters and the corresponding labels
    return np.argmin(dist, axis=1)


def update_representatives(data, clust_lab, curr_repr, clu_num):

    # Update the representatives of each cluster
    prev_repr = csr_matrix.copy(curr_repr)

    for i in range(clu_num):
        curr_repr[i] = csr_matrix(data[clust_lab == i].mean(axis=0, dtype=np.float64))

    similarity = (prev_repr - curr_repr).nnz
    if similarity == 0:
        return curr_repr, True
    else:
        return curr_repr, False


def print_results(cluster_label, t_labels, clu_num):

    results = np.zeros((clu_num, clu_num), dtype=np.float64)

    for i in range(clu_num):
        tmp = t_labels[cluster_label == i]
        for j in range(clu_num):
            results[i, j] = (len(tmp[tmp == j+1])/float(len(tmp)))*100

    # print results to csv file
    with open('clustering_KMeans.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows([[' ', 'Politics', 'Film', 'Business', 'Technology', 'Football']])
        i = 1
        for row in results:
            string = 'Cluster %d' % i
            a.writerows([[string, '{: 0.2f}'.format(row[0]), '{: 0.2f}'.format(row[1]), '{: 0.2f}'.format(row[2]), '{: 0.2f}'.format(row[3]), '{: 0.2f}'.format(row[4])]])
            i += 1

    # print results as png image
    fig = plt.figure(1)
    plt.pcolor(results, cmap='RdBu')
    plt.colorbar()
    plt.xticks(np.arange(0.5, 5.5), range(1, 6), fontsize=0)
    plt.yticks(np.arange(0.5, 5.5), range(1, 6), fontsize=15)
    plt.gca().invert_yaxis()
    plt.title(' Politics       Film     BusinessTechnologyFootball')
    plt.xlabel('Categories')
    plt.ylabel('Clusters')
    plt.show()
    fig.savefig('clustering_KMeans.png')


# Main program
# Load sparse matrix

data, labels = sdf.load_data()

data = normalize(data, norm='l2', axis=1)

# Initialize representatives according to initialization method
num_of_repr = 5
init_flag = 0

if init_flag == 0:
    representatives = rand_representatives(data, num_of_repr)
elif init_flag == 1:
    representatives = seq_representatives(data, num_of_repr)
else:
    representatives = longest_dist_representatives(data, num_of_repr)

# k-means loop
# Set 0 for euclidean, 1 for Cosine
dist_flag = 0
MAX_ITER = 100

for itr in range(MAX_ITER):
    # Calculate distances between the data and the representatives
    distances = calc_dist(dist_flag, data, representatives)

    # Calculate the clusters
    cluster_labels = update_clusters(distances)

    # Calculate the new representatives
    representatives, terminate = update_representatives(data, cluster_labels, representatives, num_of_repr)

    # terminate if the representatives don't change
    if terminate:
        break

print 'Terminate after %d iterations...'%(itr+1)
print_results(cluster_labels, labels, num_of_repr)
