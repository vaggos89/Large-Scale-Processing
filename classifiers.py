#!/usr/bin/env python

import numpy as np

# scikit-learn
from scipy.sparse import csr_matrix
import scipy.stats
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from time import time


# Main program

path = '/home/ubuntu/Desktop/Large_Scale_Tech/sparse_data_norm.npz'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/sparse_data_norm.npz'

loader = np.load(path)
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
path = '/home/ubuntu/Desktop/Large_Scale_Tech/labels_arr.npy'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/labels_arr.npy'

labels = np.load(path)

tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': [1, 1e-2, 1e-4, 1e-8], 'C': [1, 10, 100, 1000],
                    'degree': np.array(range(4)), 'coef0': [0, 1]}
                    # {'kernel': ['linear'], 'C':[1, 10, 100, 1000]},
                    # {'kernel': ['poly'], 'C':[1, 10, 100, 1000] ,'degree': np.array(range(15)), 'coef0': [0, 1]},
                    # {'kernel': ['sigmoid' ], 'C':[1, 10, 100, 1000], 'coef0': [0, 1]}]


train_d, labels_d = data[3000:7000], labels[3000:7000]
test_d, labels_test = data[2000:3000], labels[2000:3000]
print train_d.shape, labels_d.shape


clf = SVC()
start_time = time()

random_search = GridSearchCV(clf, param_grid=tuned_parameters, n_jobs=-1, cv=3)
random_search.fit(train_d, labels_d)

end_time = time()
print random_search.best_params_
# print random_search.cv_results_['mean_train_score'], random_search.cv_results_['std_train_score']
print random_search.best_score_
print 'Finish after %f'%(end_time-start_time)
'''

clf = SVC(kernel='rbf', C=1, gamma=0.0001)
clf.fit(train_d, labels_d)

# print clf.decision_function(test_d)
print (len(labels_test)-len(np.nonzero(clf.predict(test_d) - labels_test)[0]))/float(len(labels_test))
'''
