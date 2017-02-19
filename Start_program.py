#!/usr/bin/env python

import csv
import pickle
import numpy as np
import time
import store_data_format

# wordCloud
from wordcloud import STOPWORDS

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


def find_label(string):

    # convert name category to integer for faster process
    if string == 'Politics':
        return 1
    elif string == 'Film':
        return 2
    elif string == 'Business':
        return 3
    elif string == 'Technology':
        return 4
    else:
        return 5


def deconstruct_initial_data(file_name):

    # Extract from csv the fields titles, labels, texts
    global path

    lst_title = []
    lst_label = []
    lst_text = []
    lst_id = []
    with open(path+file_name, 'r') as train_csv:
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
    return lst_title, lst_text, labels_arr, lst_id

def deconstruct_test_data(file_name):

    # Extract from csv the fields titles, labels, texts
    global path

    lst_title = []
    lst_text = []
    lst_id = []
    with open(path+file_name, 'r') as train_csv:
        reader = csv.DictReader(train_csv, delimiter='\t')
        for row in reader:
            lst_title.append(row['Title'])
            lst_text.append(row['Content']+' '+row['Title'])
            lst_id.append(row['Id'])

    # Store the appropriate data for later use

    with open(path + 'text_t_lst', 'wb') as f:
        pickle.dump(lst_text, f)

    with open(path + 'id_t_lst', 'wb') as f:
        pickle.dump(lst_id, f)

    # Return values
    return lst_text, lst_id



def create_stopwords():

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



# Main Program
if __name__ == "__main__":

    file_name = 'train_set.csv'
    start_t = time.time()
    titles, texts, _, _ = deconstruct_initial_data(file_name=file_name)
    stopw = create_stopwords()
    store_data_format.generate_sparse_data(stopw, texts)


    print 'Read was completed successfully!'
    print 'Execution time (in seconds): ', time.time() - start_t
