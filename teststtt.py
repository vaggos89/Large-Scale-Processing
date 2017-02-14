import numpy as np
from time import time
import matplotlib.pyplot as plt



import pickle
# scikit-learn
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'



def Our_classifier(data, labels):

    clf1 = LogisticRegression(random_state=1, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
    clf2 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    # clf3 = SVC(decision_function_shape='ovr', C=1, gamma=1.1, probability=True)
    clf4 = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
    clfVote = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('knn', clf4)], voting='hard')

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

        train_d_l2 = train_d_l1[(train_labels_l1 == 2) | (train_labels_l1 == 1) | (train_labels_l1 == 4) | (train_labels_l1 == 3)]
        train_labels_l2 = train_labels_l1[(train_labels_l1 == 2) | (train_labels_l1 == 1) | (train_labels_l1 == 4) | (train_labels_l1 == 3)]

        train_d_l3 = train_d_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1) | (train_labels_l1 == 4)]
        train_labels_l3 = train_labels_l1[(train_labels_l1 == 3) | (train_labels_l1 == 1) | (train_labels_l1 == 4)]

        train_d_l4 = train_d_l1[(train_labels_l1 == 1) | (train_labels_l1 == 3)]
        train_labels_l4 = train_labels_l1[(train_labels_l1 == 1) | (train_labels_l1 == 3)]

        test_d, test_labels = data[test_idx], labels[test_idx]

        # Layer 1
        print 'l1'
        train_labels_l1[(train_labels_l1 == 2) | (train_labels_l1 == 1) | (train_labels_l1 == 4) | (train_labels_l1 == 3)] = 0
        test_scores[i] = clf4.fit(train_d_l1, train_labels_l1)
        predicted = clf4.predict(test_d)
        print 'l1'
        test_d = test_d[predicted == 0]
        test_labels = test_labels[predicted == 0]
        # print predicted
        if len(test_labels) == 0:
            print 'l1'
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 2
        train_labels_l2[(train_labels_l2 == 3) | (train_labels_l2 == 1) | (train_labels_l2 == 4)] = 0
        test_scores[i] = clfVote.fit(train_d_l2, train_labels_l2)
        predicted_l2 = clfVote.predict(test_d)
        print 'l1'
        predicted[predicted == 0] = predicted_l2
        test_d = test_d[predicted_l2 == 0]
        test_labels = test_labels[predicted_l2 == 0]
        # print predicted
        if len(test_labels) == 0:
            print 'l2'
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 3
        train_labels_l3[(train_labels_l3 == 3) | (train_labels_l3 == 1)] = 0
        test_scores[i] = clf1.fit(train_d_l3, train_labels_l3)
        predicted_l3 = clf1.predict(test_d)

        predicted[predicted == 0] = predicted_l3
        test_d = test_d[predicted_l3 == 0]
        test_labels = test_labels[predicted_l3 == 0]
        # print predicted
        if len(test_labels) == 0:
            print 'l3'
            accuracy[i] = accuracy_score(labels[test_idx], predicted)
            continue
        # Layer 4
        train_labels_l4[train_labels_l4 == 1] = 0
        test_scores[i] = clf1.fit(train_d_l4, train_labels_l4)
        predicted_l4 = clf1.predict(test_d)
        predicted[predicted == 0] = predicted_l4
        predicted[predicted == 0] = 1

        # test_labels[predicted_l4 == 0] = 1
        # print accuracy_score(test_labels, predicted_l4)
        # print predicted
        print 'Finish', accuracy_score(labels[test_idx], predicted)
        accuracy[i] = accuracy_score(labels[test_idx], predicted)
        i += 1
    print accuracy.mean()
        #
        # accuracy[i] = accuracy_score(test_labels[i], predicted)
        # precision[i] = precision_score(test_labels[i], predicted, average='macro', labels=l_list)
        # recall[i] = recall_score(test_labels[i], predicted, average='macro', labels=l_list)
        # f1[i] = f1_score(test_labels[i], predicted, average='macro', labels=l_list)
        # # precision[i], recall[i], f1[i], support = score(test_labels[i], predicted)
        #
        # i += 1

# Main
# Load Sparse Data
# loader = np.load(path + 'sparse_data_norm.npz')
# data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
labels = np.load(path + 'labels_arr.npy')
classifier = 0

with open(path + 'trainDataVecs', 'rb') as f:
    data = pickle.load(f)

data = normalize(data, norm='l2', axis=1)

Our_classifier(data, labels)
