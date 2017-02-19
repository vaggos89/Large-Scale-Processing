import store_data_format
import Start_program
import numpy as np
from nltk import stem
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'


def create_stem_data(texts):

    reload(sys)
    sys.setdefaultencoding('utf8')

    with open(path + 'stopwords', 'rb') as f:
            stopwords = pickle.load(f)

    vectorizer = StemmedCountVectorizer(sublinear_tf=False, min_df=5, stop_words=stopwords, norm='l2')
    data = vectorizer.fit_transform(texts)

    with open(path + 'stem_tfIdf_data_Sn', 'wb') as f:
        pickle.dump(data, f)


english_stemmer = stem.SnowballStemmer('english')


class StemmedCountVectorizer(TfidfVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


def tf_idf_data(texts_plus_titles):

    global path

    with open(path + 'stopwords', 'rb') as f:
            stopwords = pickle.load(f)

    vectorizer = TfidfVectorizer(sublinear_tf=False, min_df=5, stop_words=stopwords)
    vectorizer = vectorizer.fit(texts_plus_titles, y=[1, 2, 3, 4, 5])
    with open(path + 'tf_idf_data_vector', 'wb') as f:
        pickle.dump(vectorizer, f)

    data = vectorizer.transform(texts_plus_titles)

    print data.shape

    with open(path + 'tf_idf_data_labels', 'wb') as f:
        pickle.dump(data, f)


























file_name = 'train_set.csv'
titles, texts, _, _ = Start_program.deconstruct_initial_data(file_name=file_name)
stopw = Start_program.create_stopwords()

# store_data_format.pre_processing(titles, stopw)
store_data_format.tf_idf(texts, stopw)
# store_data_format.generate_sentences_w2v(texts, stopw)
#

# components = 1000
# flag = True
# data = store_data_format.create_store_W2V(flag, 'w2v_test_stem', components)

# _, labels = store_data_format.load_data()
# with open(path + 'snow_stem_tf_idf_data_labels', 'rb') as f:
#     data = pickle.load(f)
#
# store_data_format.create_store_SVD(True, 'tf_idf_svd_stem_labels', data, 4000)
