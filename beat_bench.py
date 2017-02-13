import numpy as np
from time import time
import matplotlib.pyplot as plt



import pickle
# scikit-learn
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


# Load Sparse Data
loader = np.load(path + 'sparse_data_norm.npz')
data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

# Load Labels array
labels = np.load(path + 'labels_arr.npy')
classifier = 0

with open(path + 'stem_tfIdf_data', 'rb') as f:
    data = pickle.load(f)

print 'Start....'
train_d, labels_d = data, labels
clf1 = LogisticRegression(random_state=1, solver='newton-cg', multi_class='multinomial', n_jobs=-1)
# clf2 = RandomForestClassifier(min_samples_leaf=3, n_estimators=150, min_samples_split=3, criterion='entropy', max_features='auto', max_depth=None, n_jobs=-1)
clf2 = RandomForestClassifier(n_estimators=150, n_jobs=-1)
clf3 = SVC(decision_function_shape='ovr', C=1, gamma=1.1, probability=True)
clf4 = GaussianNB()
clf5 = KNeighborsClassifier(n_jobs=1, n_neighbors=7)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('knn', clf5)], voting='hard', weights=[2, 1, 2])

for clf, label in zip([clf1, clf2, clf5, eclf], ['Logistic Regression', 'Random Forest', 'KNN', 'Ensemble']):
    scores = cross_val_score(clf, train_d, labels_d, cv=5, scoring='accuracy')
    print("Accuracy: %0.5f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


