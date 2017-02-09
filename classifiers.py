#!/usr/bin/env python

import numpy as np
from time import time

# scikit-learn
from scipy.sparse import csr_matrix
import scipy.stats
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def compute_auc(y_score, y_test, l_classes):

    y_test = label_binarize(y_test, classes=l_classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(l_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return roc_auc, fpr, tpr



# Main program

path = '/home/ubuntu/Desktop/Large_Scale_Tech/sparse_data_norm.npz'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/sparse_data_norm.npz'

loader = np.load(path)
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
path = '/home/ubuntu/Desktop/Large_Scale_Tech/labels_arr.npy'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/labels_arr.npy'

labels = np.load(path)
classifier = 1

if classifier == 1:
    tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'gamma': [1, 1e-2, 1e-4, 1e-8], 'C': [1, 10, 100, 1000],
                        'degree': np.array(range(4)), 'coef0': [0, 1]}
    clf = SVC()
else:
    tuned_parameters = {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [1, 3, 10],
                        "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
    clf = RandomForestClassifier()


# train_d, labels_d = data[0:6000], labels[0:6000]
# test_d, labels_test = data[9001:12000], labels[9001:12000]
#

# start_time = time()
'''
random_search = GridSearchCV(clf, param_grid=tuned_parameters, n_jobs=-1, cv=3)
random_search.fit(train_d, labels_d)

print random_search.best_params_
# print random_search.cv_results_['mean_train_score'], random_search.cv_results_['std_train_score']
print random_search.best_score_
print 'Finish after %f'%(time()-start_time)
'''

if classifier == 1:
    clf = SVC(decision_function_shape='ovr', C=1, gamma=1.1, probability=True)
else:
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)


cv_folds = 3
accuracy = np.zeros((cv_folds, 1), dtype=np.float64)
precision = np.zeros((cv_folds, 1), dtype=np.float64)
recall = np.zeros((cv_folds, 1), dtype=np.float64)
f1 = np.zeros((cv_folds, 1), dtype=np.float64)
AUC_ROC = [dict() for x in range(cv_folds)]
fpr = [dict() for x in range(cv_folds)]
tpr = [dict() for x in range(cv_folds)]
l_list = [1, 2, 3, 4, 5]
kf = KFold(n_splits=cv_folds, shuffle=True)
i = 0
for train_idx, test_idx in kf.split(data):

    train_d, train_labels = data[train_idx], labels[train_idx]
    test_d, test_labels = data[test_idx], labels[test_idx]

    # start_time = time()
    test_scores = clf.fit(train_d, train_labels).decision_function(test_d)

    # print clf.decision_function(test_d)
    # predicted = cross_val_predict(clf, train_d, labels_d, n_jobs=-1, cv=10)
    predicted = clf.predict(test_d)
    accuracy[i] = accuracy_score(test_labels, predicted)
    precision[i] = precision_score(test_labels, predicted, average='micro', labels=l_list)
    recall[i] = recall_score(test_labels, predicted, average='micro', labels=l_list)
    f1[i] = f1_score(test_labels, predicted, average='micro', labels=l_list)
    AUC_ROC[i], fpr[i], tpr[i] = compute_auc(y_score=test_scores, y_test=test_labels, l_classes=l_list)
    # roc_auc_score(recall, pres, average='micro')
    i += 1
    print i

print 'accuracy_score =', accuracy.mean()
print 'precision_score=', precision.mean()
print 'recall_score   =', recall.mean()
print 'f1_score       =', f1.mean()

# print roc_auc_score(recall, pres, average='micro')
# print roc_curve(labels, predicted)
# print classification_report(test_labels, predicted, target_names=['1', '2', '3', '4', '5'])
