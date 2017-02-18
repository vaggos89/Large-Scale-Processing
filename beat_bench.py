import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
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
from sklearn.linear_model import SGDClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
# path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
path = '/home/apostolis/Desktop/Large_Scale_Tech/'

from sklearn.model_selection import cross_val_predict
# Load Sparse Data
data, labels = sdf.load_data()

with open(path + 'tf_idf_data_labels', 'rb') as f:
    data = pickle.load(f)

print data.shape
print 'Start....'

# (labels == 1) | (labels == 2) | (labels == 3) | (labels == 4)
# (labels == 1) | (labels == 2) | (labels == 3)
# (labels == 1) | (labels == 2)

# (labels_d == 1) | (labels_d == 2) | (labels_d == 3) | (labels_d == 4)
# (labels_d == 1) | (labels_d == 2) | (labels_d == 3)
# (labels_d == 1) | (labels_d == 2)

train_d = data[(labels == 1) | (labels == 3)]
labels_d = labels[(labels == 1) | (labels == 3)]

labels_d[(labels_d == 1)] = 1
labels_d[(labels_d == 3)] = 0

clf1 = LogisticRegression(random_state=1, solver='newton-cg', class_weight="balanced", warm_start=False)
# clf2 = RandomForestClassifier(min_samples_leaf=3, n_estimators=150, min_samples_split=3, criterion='entropy', max_features='auto', max_depth=None, n_jobs=-1)
clf4 = SGDClassifier(loss='hinge', random_state=1, class_weight="balanced", n_iter=5, penalty='l2', n_jobs=1)
clf5 = KNeighborsClassifier(n_neighbors=5)

eclf = VotingClassifier(estimators=[('lr', clf1), ('SGD', clf4), ('kNN', clf5)], voting='hard', weights=[2, 1.5, 1])

st = time()

# scores = cross_val_score(clf1, train_d, labels_d, cv=5, scoring='accuracy', n_jobs=-1)
# print("Accuracy: %0.5f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), 'ensemble'))

for i in range(4):
    for clf, label in zip([clf1, clf4, clf5, eclf], ['Logistic Regression', 'SGD', 'kNN', 'Ensemble']):
        scores = cross_val_score(clf, train_d, labels_d, cv=KFold(n_splits=10, shuffle=True), scoring='accuracy',n_jobs=-1)
        print("Accuracy: %0.5f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


print 'Elapsed time: ', time() - st
