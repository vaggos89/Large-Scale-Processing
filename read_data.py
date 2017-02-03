#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt

# wordCloud
from wordcloud import WordCloud, STOPWORDS

# convert name category to integer for faster process


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
file_name = '/home/ubuntu/Desktop/Large_Scale_Tech/train_set.csv'
#file_name = '/home/apostolis/Desktop/Large_Scale_Tech/train_set.csv'

titles, labels, texts, ids = separate_cat_text(file_name)

'''
for i in range(5):
    string = 'text_%d.txt'%(i+1)
    target = open(string, 'w')

    j = 0
    # idx = np.where(labels == i+1)
    # tmp = idx[0][:]
    # test = texts.__getitem__(tmp)
    # print(test)
    for text in texts:
        if labels[j] == i+1:
            target.write(text)
        j += 1
    target.close()

'''
stopf = open('ranksnl_stopwords.txt')
stopwords = stopf.read()
stopwords = stopwords.split('\n')
stopf.close()

stopwords = set(stopwords)
stopw = set(STOPWORDS)
stopwords = stopwords.union(stopw)

for i in range(5):

    string = '/home/ubuntu/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)
    #string = '/home/apostolis/Desktop/Large_Scale_Tech/text_%d.txt'%(i+1)

    text = open(string).read()

    wordcloud = WordCloud(background_color='white', scale=1, width=800, height=600, stopwords=stopwords, max_words=200).generate(text)

    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    string = '/home/ubuntu/Desktop/Large_Scale_Tech/WordCloud_%d.tiff'%(i+1)
    # string = '/home/apostolis/Desktop/Large_Scale_Tech/WordCloud_%d.tiff'%(i+1)
    plt.savefig(string)
    # plt.show()
    plt.close()
