#!/usr/bin/env python

import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# wordCloud
from wordcloud import WordCloud, STOPWORDS


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

    path = '/home/ubuntu/Desktop/Large_Scale_Tech/'
    # path = '/home/apostolis/Desktop/Large_Scale_Tech/'

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

    path = '/home/ubuntu/Desktop/Large_Scale_Tech/train_set.csv'
    # path = '/home/apostolis/Desktop/Large_Scale_Tech/train_set.csv'

    # Separate data from csv
    titles, labels, texts, ids = separate_cat_text(path)

    # For each category generate a file with corresponding contents
    for i in range(5):

        path = '/home/ubuntu/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)
        # path = '/home/apostolis/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)
        target = open(path, 'w')

        j = 0
        for text in texts:
            if labels[j] == i+1:
                target.write(text)
            j += 1
        target.close()

    return titles, texts


def create_wordcloud():

    # Create a wordcloud for each category

    path = '/home/ubuntu/Desktop/Large_Scale_Tech/ranksnl_stopwords.txt'
    # path = '/home/apostolis/Desktop/Large_Scale_Tech/ranksnl_stopwords.txt'

    f = open(path)
    stopwords1 = f.read()
    stopwords1 = stopwords1.split('\n')
    stopwords1 = set(stopwords1)
    f.close()

    stopwords2 = set(STOPWORDS)
    stopwords = stopwords1.union(stopwords2)

    path = '/home/ubuntu/Desktop/Large_Scale_Tech/stopwords'
    # path = '/home/apostolis/Desktop/Large_Scale_Tech/stopwords'

    with open(path, 'wb') as f:
        pickle.dump(stopwords, f)

    for i in range(5):

        path = '/home/ubuntu/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)
        # path = '/home/apostolis/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)

        text = open(path).read()

        wordcloud = WordCloud(background_color='white', scale=1, width=800, height=600, stopwords=stopwords, max_words=200).generate(text)

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        path = '/home/ubuntu/Desktop/Large_Scale_Tech/WordCloud_%d.tiff'%(i+1)
        # path = '/home/apostolis/Desktop/Large_Scale_Tech/WordCloud_%d.tiff'%(i+1)
        plt.savefig(path)
        # plt.show()
        plt.close()

    return stopwords


def generate_sparse_data(stopwords, texts_plus_titles):

    vectorizer = CountVectorizer(analyzer='word', stop_words=stopwords)
    sparse_data = vectorizer.fit_transform(texts_plus_titles)

    # normalize data
    sparse_data_norm = normalize(sparse_data, norm='l2', axis=1)

    # save sparse matrix
    path = '/home/ubuntu/Desktop/Large_Scale_Tech/sparse_data_norm'
    # path = '/home/apostolis/Desktop/Large_Scale_Tech/sparse_data_norm'

    np.savez(path, data=sparse_data_norm.data, indices=sparse_data_norm.indices, indptr=sparse_data_norm.indptr, shape=sparse_data_norm.shape )
    # np.savez('X_array_norm', data=sA.data, indices=sA.indices, indptr=sA.indptr, shape=sA.shape )


# Main Program

print 'Preprocessing the data..'

start_t = time.time()

titles, texts = deconstruct_initial_data()
stopw = create_wordcloud()
generate_sparse_data(stopw, texts)

end_t = time.time()

print 'Preprocessing was completed successfully!'
print 'Execution time (in seconds): ', end_t - start_t
