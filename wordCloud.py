import store_data_format
import read_data_v2

titles, texts = read_data_v2.deconstruct_initial_data()
stopw = read_data_v2.create_stopwords()
store_data_format.create_wordcloud(stopw)
