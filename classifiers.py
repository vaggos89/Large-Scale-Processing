#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from scipy import interp
from itertools import cycle
import gensim
from time import time

# scikit-learn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# path = '/home/ubuntu/Desktop/Large_Scale_Tech/'
path = '/home/apostolis/Desktop/Large_Scale_Tech/'


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
def plot_roc_auc(fpr, tpr, roc_auc, n_classes):

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
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')
    plt.show()


# Main program

# Load Sparse Data
loader = np.load(path + 'sparse_data_norm.npz')
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
labels = np.load(path + 'labels_arr.npy')

# Choose classifier
# {0}:Random Forests (RFC)
# {1}:SVM
classifier = 0

if classifier == 1:
    clf = SVC(decision_function_shape='ovr', C=1, gamma=1.1)
else:
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)

# Choose the feature method
# {0}:Bag of Words                  (BoW)
# {1}:Singular value decomposition  (SVD)
# {2}:Word 2 Vect                   (W2V)
feat_method = 2

if feat_method == 1:
    svd = TruncatedSVD(n_components=5000, n_iter=7)
    data = svd.fit_transform(data)

    # print(svd.explained_variance_ratio_)
    print(svd.explained_variance_ratio_.sum())
elif feat_method == 2:

    with open(path + 'sentences', 'rb') as f:
        sentences = pickle.load(f)

    model = gensim.models.Word2Vec(sentences, min_count=5, size=10, workers=4)

    model.save_word2vec_format(path + 'model_v1', binary=True)

    model.train(sentences)

    sys.exit(0)
else:
    print 'BoW'

# Choose K_folds cross validation
cv_folds = 10

test_scores = [None]*cv_folds
test_labels = [None]*cv_folds

accuracy = np.zeros((cv_folds, 1), dtype=np.float64)
precision = np.zeros((cv_folds, 1), dtype=np.float64)
recall = np.zeros((cv_folds, 1), dtype=np.float64)
f1 = np.zeros((cv_folds, 1), dtype=np.float64)

l_list = [1, 2, 3, 4, 5]

kf = KFold(n_splits=cv_folds, shuffle=True)
i = 0

for train_idx, test_idx in kf.split(data):

    train_d, train_labels = data[train_idx], labels[train_idx]
    test_d, test_labels[i] = data[test_idx], labels[test_idx]

    test_scores[i] = clf.fit(train_d, train_labels).predict_proba(test_d)
    predicted = clf.predict(test_d)

    accuracy[i] = accuracy_score(test_labels[i], predicted)
    precision[i] = precision_score(test_labels[i], predicted, average='micro', labels=l_list)
    recall[i] = recall_score(test_labels[i], predicted, average='micro', labels=l_list)
    f1[i] = f1_score(test_labels[i], predicted, average='micro', labels=l_list)

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
print 'AUC            =', AUC_ROC['micro']


