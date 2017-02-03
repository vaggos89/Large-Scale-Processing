#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
#Sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer
#wordCloud
from wordcloud import WordCloud, STOPWORDS
#from nltk.corpus import stopwords
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
	else :
		label = 5

	return label

# extract from csv the fields titles, labels, texts
def seperate_cat_text(file_name):
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
		return 	lst_title, labels, lst_text, lst_id



# Main program
file_name = 'newtrain_set.csv'

titles, labels, texts, ids = seperate_cat_text(file_name)
'''
for i in range(5):
	string = 'text_%d.txt'%(i+1)
	target = open(string, 'w')
	j = 0
	for text in texts:
		if labels[j] == i+1:
			target.write(text)
		j = j + 1
	target.close()

with open('stopwords.txt', 'wb') as f:
	pickle.dump(stopwords, f)

with open('stopwords.txt', 'rb') as f:
	stopwords = pickle.load(f)
'''
stop = set(stopwords.words('english'))
#stopwords = set(stopwords)

text = open('text_2.txt').read()
#parameter for wordcloud
wordcloud = WordCloud(background_color="white", stopwords=stop).generate(text)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('/home/ubuntu/Desktop/Large_Scale_Tech/WordCloud_1.tiff')
plt.show()
plt.close()

#vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english')
#X = vectorizer.fit_transform(texts)