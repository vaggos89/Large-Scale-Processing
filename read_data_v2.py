#!/usr/bin/env python

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import re

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# wordCloud
from wordcloud import WordCloud, STOPWORDS

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'

def find_label(string):

    # convert name category to integer for faster process

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


def separate_cat_text(file_name):

    # Extract from csv the fields titles, labels, texts

    global path

    lst_title = []
    lst_label = []
    lst_text = []
    lst_id = []
    with open(file_name, 'r') as train_csv:
        reader = csv.DictReader(train_csv, delimiter='\t')
        for row in reader:
            lst_title.append(row['Title'])
            lst_label.append(find_label(row['Category']))
            lst_text.append(row['Content']+' '+row['Title'])
            lst_id.append(row['Id'])

    labels_arr = np.array([x for x in lst_label])

    # Store the appropriate data for later use

    with open(path + 'titles_lst', 'wb') as f:
        pickle.dump(lst_title, f)

    with open(path + 'text_lst', 'wb') as f:
        pickle.dump(lst_text, f)

    with open(path + 'id_lst', 'wb') as f:
        pickle.dump(lst_id, f)

    np.save(path + 'labels_arr', labels_arr)

    # Return values
    return lst_title, labels_arr, lst_text, lst_id


def deconstruct_initial_data():

    # Generate the appropriate data files and store for later use

    global path

    # Separate data from csv
    titles, labels, texts, ids = separate_cat_text(path + 'train_set.csv')

    # For each category generate a file with corresponding contents
    for i in range(5):

        ext = 'text_%d.txt'%(i+1)

        target = open(path + ext, 'w')

        j = 0
        for text in texts:
            if labels[j] == i+1:
                target.write(text)
            j += 1
        target.close()

    return titles, texts


def create_stopwords():

    global path

    f = open(path + 'ranksnl_stopwords.txt')
    stopwords1 = f.read()
    stopwords1 = stopwords1.split('\n')
    stopwords1 = set(stopwords1)
    f.close()

    stopwords2 = set(STOPWORDS)
    stopwords = stopwords1.union(stopwords2)

    with open(path + 'stopwords', 'wb') as f:
        pickle.dump(stopwords, f)

    return stopwords


def create_wordcloud(stopwords):

    # Create a wordcloud for each category
    for i in range(5):

        ext = 'text_%d.txt'%(i+1)

        text = open(path + ext).read()

        wordcloud = WordCloud(background_color='white', scale=1, width=800, height=600, stopwords=stopwords, max_words=200).generate(text)

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        ext = 'WordCloud_%d.tiff'%(i+1)
        plt.savefig(path + ext)
        # plt.show()
        plt.close()


def generate_sparse_data(stopwords, texts_plus_titles):

    global path

    vectorizer = CountVectorizer(analyzer='word', stop_words=stopwords)
    sparse_data = vectorizer.fit_transform(texts_plus_titles)

    # normalize data
    sparse_data_norm = normalize(sparse_data, norm='l2', axis=1)

    # save sparse matrix
    np.savez(path + 'sparse_data_norm', data=sparse_data_norm.data, indices=sparse_data_norm.indices, indptr=sparse_data_norm.indptr, shape=sparse_data_norm.shape )
    # np.savez('X_array_norm', data=sA.data, indices=sA.indices, indptr=sA.indptr, shape=sA.shape )


def generate_sentences_w2v(texts, stopwords):

    global path

    sentences = [[word for word in re.split('[. ;!?,]', document.lower()) if word not in stopwords] for document in texts]

    with open(path + 'sentences', 'wb') as f:
        pickle.dump(sentences, f)


# Main Program
print 'Preprocessing the data..'

start_t = time.time()

titles, texts = deconstruct_initial_data()
stopw = create_stopwords()
# create_wordcloud(stopw)
# generate_sparse_data(stopw, texts)
generate_sentences_w2v(texts, stopw)

end_t = time.time()

print 'Preprocessing was completed successfully!'
print 'Execution time (in seconds): ', end_t - start_t
