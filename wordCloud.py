import read_data_v2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import numpy as np
from itertools import compress


path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
# path = '/home/apostolis/Desktop/Large_Scale_Tech/'


def create_wordcloud(stopwords):

    with open(path + 'text_lst', 'rb') as f:
        texts = pickle.load(f)

    labels = np.load(path + 'labels_arr.npy')

    # Create a wordcloud for each category
    for i in range(5):

        # ext = 'text_%d.txt'%(i+1)
        # text = open(path + ext).read()

        wordcloud = WordCloud(background_color='white', scale=1, width=800, height=600, stopwords=stopwords, max_words=200).generate(''.join(list(compress(texts, labels==i+1))))

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        ext = 'WordCloud_%d.tiff'%(i+1)
        plt.savefig(path + ext)
        # plt.show()
        plt.close()


# Main
with open(path + 'stopwords', 'rb') as f:
        stopw = pickle.load(f)


create_wordcloud(stopw)
print 'Word Clouds Ready...'
