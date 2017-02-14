#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import store_data_format as sdf

# scikit-learn
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


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


# Imported from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def plot_roc_auc(fpr, tpr, roc_auc, n_classes, classifier, method):

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black', 'green'])
    lw = 2

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                 ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc='best')
    string = 'ROC_%d_%d.png'%(classifier, method)
    plt.savefig(string)
    # plt.show()


# Main program
# [Store Data]->flag==True
# [Load Data ]->flag==False
flag = False

# Choose the feature method ############
# {0}:Bag of Words                  (BoW)
# {1}:Singular value decomposition  (SVD)
# {2}:Word 2 Vector                 (W2V)
for feat_method in [0, 1, 2]:

    if feat_method == 1:
        data, labels = sdf.load_data()
        f_name = 'svd_data_d4000'
        components = 4000
        sdf.create_store_SVD(flag, f_name, data, components)

    elif feat_method == 2:
        f_name = 'w2v_data'
        components = 128
        labels = np.load(path + 'labels_arr.npy')
        data = sdf.create_store_W2V(flag, f_name, components)
    else:
        data, labels = sdf.load_data()
        print 'Feature Method:BoW...'

    # Choose K_folds cross validation
    cv_folds = 10

    test_scores = [None]*cv_folds
    test_labels = [None]*cv_folds

    accuracy = np.zeros((cv_folds, 2, 3), dtype=np.float64)
    precision = np.zeros((cv_folds, 2, 3), dtype=np.float64)
    recall = np.zeros((cv_folds, 2, 3), dtype=np.float64)
    f1 = np.zeros((cv_folds, 2, 3), dtype=np.float64)
    l_list = [1, 2, 3, 4, 5]

    kf = KFold(n_splits=cv_folds, shuffle=True)
    i = 0

    print 'Compute Results...\n'
    for train_idx, test_idx in kf.split(data):
        # Choose classifier ############
        # {0}:SVM
        # {1}:Random Forests (RFC)
        for classifier in [0, 1]:
            if classifier == 0:
                clf = SVC(decision_function_shape='ovr', kernel='rbf', C=1, gamma=1, probability=True)
                print 'Classifier    :SVC...'
            else:
                clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
                print 'Classifier    :RFC...'

            train_d, train_labels = data[train_idx], labels[train_idx]
            test_d, test_labels[i] = data[test_idx], labels[test_idx]

            test_scores[i] = clf.fit(train_d, train_labels).predict_proba(test_d)
            predicted = clf.predict(test_d)

            accuracy[i, classifier, feat_method] = accuracy_score(test_labels[i], predicted)
            precision[i, classifier, feat_method] = precision_score(test_labels[i], predicted, average='macro', labels=l_list)
            recall[i, classifier, feat_method] = recall_score(test_labels[i], predicted, average='macro', labels=l_list)
            f1[i, classifier, feat_method] = f1_score(test_labels[i], predicted, average='macro', labels=l_list)

        i += 1
    # Calculate AUC and ROC plot
    all_test_scores = np.concatenate([x for x in test_scores])
    all_test_labels = np.concatenate([x for x in test_labels])

    AUC_ROC, fpr, tpr = compute_auc(y_score=all_test_scores, y_test=all_test_labels, l_classes=l_list)
    plot_roc_auc(fpr, tpr, AUC_ROC, len(l_list))

    print 'accuracy_score =', accuracy.mean()
    print 'precision_score=', precision.mean()
    print 'recall_score   =', recall.mean()
    print 'f1_score       =', f1.mean()
    print 'AUC            =', AUC_ROC['macro']


