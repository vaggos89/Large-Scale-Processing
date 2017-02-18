import numpy as np
from time import time
import matplotlib.pyplot as plt

import Start_program
import store_data_format as sdf

import pickle
# scikit-learn
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


def my_method(data, labels):

    clf1 = LogisticRegression(random_state=1, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=150, n_jobs=-1)
    # clf3 = SVC(decision_function_shape='ovr', C=1, gamma=1.1, probability=True)
    clf4 = KNeighborsClassifier(n_jobs=1, n_neighbors=5)
    clf5 = MultinomialNB()
    clf6 = SGDClassifier(loss='hinge', n_iter=5, penalty='l2', n_jobs=-1)
    clfVote = VotingClassifier(estimators=[('lr', clf1), ('nd', clf5), ('sgd', clf6), ('knn', clf4)], voting='hard', weights=[2, 1, 1.5, 1])
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

        train_d_l1, train_labels_l1 = data[train_idx], labels[train_idx]

        train_d_l2 = train_d_l1[(train_labels_l1 == 3) | (train_labels_l1 == 2) | (train_labels_l1 == 4) | (train_labels_l1 == 1)]
        train_labels_l2 = train_labels_l1[(train_labels_l1 == 3) | (train_labels_l1 == 2) | (train_labels_l1 == 4) | (train_labels_l1 == 1)]

        train_d_l3 = train_d_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1) | (train_labels_l1 == 4)]
        train_labels_l3 = train_labels_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1) | (train_labels_l1 == 4)]

        train_d_l4 = train_d_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1)]
        train_labels_l4 = train_labels_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1)]

        test_d, test_labels = data[test_idx], labels[test_idx]

        # Layer 1
        train_labels_l1[(train_labels_l1 == 3) | (train_labels_l1 == 2) | (train_labels_l1 == 4) | (train_labels_l1 == 1)] = 0
        test_scores[i] = clf4.fit(train_d_l1, train_labels_l1)
        predicted = clf4.predict(test_d)

        test_d = test_d[predicted == 0]
        test_labels = test_labels[predicted == 0]
        # print predicted
        if len(test_labels) == 0:
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 2
        train_labels_l2[(train_labels_l2 == 3) | (train_labels_l2 == 1) | (train_labels_l2 == 4)] = 0
        test_scores[i] = clf6.fit(train_d_l2, train_labels_l2)
        predicted_l2 = clf6.predict(test_d)

        predicted[predicted == 0] = predicted_l2
        test_d = test_d[predicted_l2 == 0]
        test_labels = test_labels[predicted_l2 == 0]
        # print predicted
        if len(test_labels) == 0:
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 3
        train_labels_l3[(train_labels_l3 == 1) | (train_labels_l3 == 3)] = 0
        test_scores[i] = clf4.fit(train_d_l3, train_labels_l3)
        predicted_l3 = clf4.predict(test_d)

        predicted[predicted == 0] = predicted_l3
        test_d = test_d[predicted_l3 == 0]
        test_labels = test_labels[predicted_l3 == 0]
        # print predicted
        if len(test_labels) == 0:
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 4
        train_labels_l4[train_labels_l4 == 3] = 0
        test_scores[i] = clfVote.fit(train_d_l4, train_labels_l4)
        predicted_l4 = clfVote.predict(test_d)
        predicted[predicted == 0] = predicted_l4
        predicted[predicted == 0] = 3

        # test_labels[predicted_l4 == 0] = 1
        # print accuracy_score(test_labels, predicted_l4)
        # print predicted
        # print 'Finish', accuracy_score(labels[test_idx], predicted)
        accuracy[i] = accuracy_score(labels[test_idx], predicted)
        i += 1
    return accuracy.mean()



data, labels = sdf.load_data()
data = normalize(data, norm='l2', axis=1)

with open(path + 'tf_idf_data', 'rb') as f:
    data = pickle.load(f)


print data.shape

# data = SelectKBest(chi2, k=20000).fit_transform(data, labels)
acc = np.zeros(4, dtype=np.float64)
print data.shape
for i in [1, 2, 3, 0]:
    acc[i] = my_method(data, labels)
print acc.mean()
