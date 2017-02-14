#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from scipy import interp
from itertools import cycle
from gensim.models import Word2Vec, word2vec
import logging
from time import time

# scikit-learn
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

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


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype=np.float64)
    #
    nwords = 0
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)

    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype=np.float64)
    #
    # Loop through the reviews
    for review in reviews:

        # Print a status message every 1000th review
        if counter%1000 == 0:
            print "Review %d of %d" % (counter, len(reviews))

        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

        # Increment the counter
        counter += 1
    return reviewFeatureVecs

# Main program

# Load Sparse Data
loader = np.load(path + 'sparse_data_norm.npz')
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
labels = np.load(path + 'labels_arr.npy')

# DeBug mode for Store or Load
deBug = False

# Choose classifier ############
# {0}:Random Forests (RFC)
# {1}:SVM
classifier = 2
if classifier == 1:
    clf = SVC(decision_function_shape='ovr', kernel='rbf', C=1, gamma=1, probability=True)
    print 'Classifier    :SVC...'

else:
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    print 'Classifier    :RFC...'

# Choose the feature method ############
# {0}:Bag of Words                  (BoW)
# {1}:Singular value decomposition  (SVD)
# {2}:Word 2 Vector                 (W2V)
feat_method = 2
if feat_method == 1:

    if deBug:
        svd = TruncatedSVD(n_components=4000, n_iter=5)
        data = svd.fit_transform(data)
        # print(svd.explained_variance_ratio_)
        print(svd.explained_variance_ratio_.sum())
        with open(path + 'svd_data', 'wb') as f:
                pickle.dump(data, f)
    else:
        with open(path + 'svd_data_d4000', 'rb') as f:
            data = pickle.load(f)
    print 'Feature Method:SVD...'


elif feat_method == 2:

    if deBug:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        with open(path + 'sentences', 'rb') as f:
            sentences = pickle.load(f)
        dim = 1024
        model = word2vec.Word2Vec(sentences, min_count=5, size=dim, workers=8)
        model.init_sims(replace=True)

        data = getAvgFeatureVecs(sentences, model, num_features=dim)

        model.save_word2vec_format(path + 'model_v1.model.bin', binary=True)
        # model = word2vec.Word2Vec.load_word2vec_format(path + 'model_v1.model.bin', binary=True)
        with open(path + 'trainDataVecs', 'wb') as f:
            pickle.dump(data, f)
        sys.exit(0)
    else:
        with open(path + 'trainDataVecs', 'rb') as f:
            data = pickle.load(f)
    print 'Feature Method:W2V...'
    # plt.plot(data[:, 1], data[:, 0], 'r*')
    # plt.show()

else:

    print 'Feature Method:BoW...'

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

# data_test = data[10001:-1]
# labels_test = labels[10001:-1]
# data = data[0:10000]
# labels = labels[0:10000]

print 'Compute Results...\n'

for train_idx, test_idx in kf.split(data):

    train_d, train_labels = data[train_idx], labels[train_idx]
    test_d, test_labels[i] = data[test_idx], labels[test_idx]

    test_scores[i] = clf.fit(train_d, train_labels).predict_proba(test_d)
    predicted = clf.predict(test_d)

    accuracy[i] = accuracy_score(test_labels[i], predicted)
    precision[i] = precision_score(test_labels[i], predicted, average='macro', labels=l_list)
    recall[i] = recall_score(test_labels[i], predicted, average='macro', labels=l_list)
    f1[i] = f1_score(test_labels[i], predicted, average='macro', labels=l_list)
    # precision[i], recall[i], f1[i], support = score(test_labels[i], predicted)

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


