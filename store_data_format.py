#!/usr/bin/env python

import pickle
import numpy as np
import re
from nltk import stem
import sys
from scipy.sparse import csr_matrix
import logging

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from gensim.models import word2vec

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


def generate_sparse_data(stopwords, texts_plus_titles):

    vectorizer = CountVectorizer(analyzer='word', min_df=1, stop_words=stopwords)
    sparse_data = vectorizer.fit_transform(texts_plus_titles)

    sparse_data = normalize(sparse_data, norm='l2', axis=1)

    # save sparse matrix
    np.savez(path + 'sparse_data', data=sparse_data.data, indices=sparse_data.indices, indptr=sparse_data.indptr, shape=sparse_data.shape )


def generate_sentences_w2v(texts, stopwords):

    global path

    sentences = [[word for word in re.split('[. ;!?,]', document.lower()) if word not in stopwords] for document in texts]

    with open(path + 'sentences', 'wb') as f:
        pickle.dump(sentences, f)


def pre_processing(texts, stopwords):

    reload(sys)
    sys.setdefaultencoding('utf8')

    vectorizer = StemmedCountVectorizer(sublinear_tf=True, min_df=5, stop_words=stopwords, norm='l2')
    data = vectorizer.fit_transform(texts)

    with open(path + 'stem_tfIdf_data_Sn', 'wb') as f:
        pickle.dump(data, f)


english_stemmer = stem.SnowballStemmer('english')


class StemmedCountVectorizer(TfidfVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def tf_idf(texts_plus_titles, stopwords):

    global path

    vectorizer = TfidfVectorizer(sublinear_tf=False, min_df=5, stop_words=stopwords)
    vectorizer = vectorizer.fit(texts_plus_titles, y=[1, 2, 3, 4, 5])
    with open(path + 'tf_idf_data_vector', 'wb') as f:
        pickle.dump(vectorizer, f)

    data = vectorizer.transform(texts_plus_titles)


    print data.shape

    with open(path + 'tf_idf_data_labels', 'wb') as f:
        pickle.dump(data, f)


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


def create_store_SVD(deBug, filename, data, components):

    if deBug:
        svd = TruncatedSVD(n_components=components, n_iter=5)
        data = svd.fit_transform(data)

        print 'Variance_Ratio:', svd.explained_variance_ratio_.sum()

        with open(path + filename, 'wb') as f:
                pickle.dump(data, f)
        return data

    else:

        with open(path + filename, 'rb') as f:
            data = pickle.load(f)
        return data


def create_store_W2V(deBug, filename, components):

    # reload(sys)
    # sys.setdefaultencoding('utf8')
    if deBug:

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        with open(path + 'sentences', 'rb') as f:
            sentences = pickle.load(f)

        # english_stemmer = stem.SnowballStemmer('english')
        # sentences_stem = []
        # i = 0
        # for doc in sentences:
        #     sentences_stem.append([english_stemmer.stem(w) for w in doc])
        # print len(sentences_stem)
        # # sys.exit(0)
        model = word2vec.Word2Vec(sentences, min_count=5, size=components, workers=8)
        model.init_sims(replace=True)

        data = getAvgFeatureVecs(sentences, model, num_features=components)

        model.save_word2vec_format(path + 'model_v1.model.bin', binary=True)

        with open(path + filename, 'wb') as f:
            pickle.dump(data, f)

        return data
    else:
        with open(path + filename, 'rb') as f:
            data = pickle.load(f)

        return data


def load_data():
    # Load Sparse Data
    loader = np.load(path + 'sparse_data.npz')
    data = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    # Load Labels array
    labels = np.load(path + 'labels_arr.npy')

    return data, labels
