import csv
import numpy as np
import matplotlib.pyplot as plt

#scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

from wordcloud import STOPWORDS


def find_label(string):

    if string == 'Politics':
        label = 1
    elif string == 'Film':
        label = 2
    elif string == 'Business':
        label = 3
    elif string == 'Technology':
        label = 4
    else:
        label = 5

    return label

# extract from csv the fields titles, labels, texts


def separate_cat_text(file_name):

    lst_title = []
    lst_label = []
    lst_text = []
    lst_id = []
    with open(file_name, 'r') as train_csv:
        reader = csv.DictReader(train_csv, delimiter='\t')
        for row in reader:
            lst_title.append(row['Title'])
            lst_label.append(find_label(row['Category']))
            lst_text.append(row['Content'])
            lst_id.append(row['Id'])

    labels = np.array([x for x in lst_label])

    return lst_title, labels, lst_text, lst_id


# Main program
# file_name = '/home/ubuntu/Desktop/Large_Scale_Tech/train_set.csv'
file_name = '/home/apostolis/Desktop/Large_Scale_Tech/train_set.csv'

flag = 0
titles, labels, texts, ids = separate_cat_text(file_name)


if flag == 1:
    stopf = open('ranksnl_stopwords.txt')
    stopwords = stopf.read()
    stopwords = stopwords.split('\n')
    stopf.close()

    stopwords = set(stopwords)
    stopw = set(STOPWORDS)
    stopwords = stopwords.union(stopw)
    texts_plus_titles = list(set(texts+titles))

    vectorizer = CountVectorizer(analyzer='word', stop_words=stopwords)
    X_array = vectorizer.fit_transform(texts_plus_titles)
    # print X.shape
    # X_array = X.transpose()
    # use sparse matrix for processing because the array is to big to fit in memory
    sA = csr_matrix(X_array)
    X_array_norm = normalize(sA, norm='l2', axis=1)
    # save sparse matrix
    np.savez('X_array_norm', data=X_array_norm.data, indices=X_array_norm.indices, indptr=X_array_norm.indptr, shape=X_array_norm.shape )
    # np.savez('X_array_norm', data=sA.data, indices=sA.indices, indptr=sA.indptr, shape=sA.shape )
    print 'Save data is done...'

else:
    # load sparse matrix
    loader = np.load('X_array_norm.npz')
    X_array_norm = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    print 'Load data is done...'

# print test
print 'Shape =', X_array_norm.shape
