import store_data_format
import Start_program
import numpy as np
import pickle

path = '/media/ubuntu/FAD42D9DD42D5CDF/Master/Lessons/Large_Scale_Tech/'
#
# file_name = 'train_set.csv'
# titles, texts, _, _ = Start_program.deconstruct_initial_data(file_name=file_name)
# stopw = Start_program.create_stopwords()

# store_data_format.pre_processing(titles, stopw)
# store_data_format.tf_idf(texts, stopw)
# store_data_format.generate_sentences_w2v(texts, stopw)
#

# components = 1000
# flag = True
# data = store_data_format.create_store_W2V(flag, 'w2v_test_stem', components)

_, labels = store_data_format.load_data()
with open(path + 'tf_idf_data', 'rb') as f:
    data = pickle.load(f)

data = data[(labels == 3) | (labels == 1)]

store_data_format.create_store_SVD(True, 'tf_idf_svd_1_3', data, 5000)
