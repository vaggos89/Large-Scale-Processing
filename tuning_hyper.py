#!/usr/bin/env python


import numpy as np
from time import time
import matplotlib.pyplot as plt


# scikit-learn
from scipy.sparse import csr_matrix
import scipy.stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


# Load Sparse Data
loader = np.load(path + 'sparse_data_norm.npz')
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
labels = np.load(path + 'labels_arr.npy')
classifier = 0

train_d, labels_d = data[0:2000], labels[0:2000]
test_d, labels_test = data[9001:12000], labels[9001:12000]


# plt.hist(labels, bins='auto')
# plt.show()

for classifier in range(2):
    if classifier == 1:
        tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': [2, 1.5, 1, 1e-2, 1e-4], 'C': [0.1, 0.5, 1, 5, 10],
                            'degree': np.array(range(4)), 'coef0': [0, 1]}
        clf = SVC()
    else:
        tuned_parameters = {'n_estimators': [10, 50, 100], 'max_depth': [3, None], 'max_features': ['auto', 'log2', 0.3, 0.5], 'min_samples_split': [2, 3, 10],
                            'min_samples_leaf': [1, 3, 10], 'bootstrap': [True, False], 'criterion': ["gini", "entropy"]}
        clf = RandomForestClassifier()

    start_time = time()
    random_search = RandomizedSearchCV(clf, n_iter=10, param_distributions=tuned_parameters, n_jobs=-1)
    random_search.fit(train_d, labels_d)

    print random_search.best_params_
    print random_search.best_score_
    print 'Finish after %f'%(time()-start_time)
