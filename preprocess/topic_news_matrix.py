import os
import sys
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
wordvec_path = script_path + 'word_vector/'
data_path = package_path + 'data/'
utils_path = package_path + 'utils/'

sys.path.append(wordvec_path)
sys.path.append(script_path)
sys.path.append(utils_path)

from mat import *
from word_vector import WordVector
from news2vec import News2Vec


"""
create topic doc similarity matrix

"""


class TopicNewsMatrix:
    def __init__(self, config):
        self.config = config

        self.wordvec = WordVector(model=config['word_vector']['model'])
        self.wordvec.restore()

        self.id2topic = dict()

    def create(self):
        """
        sim_mat axis0 - news_id
                axis1 - topic_id
        """

        news_mat = self.__get_news_matrix()
        topic_mat = self.get_topic_matrix()

        print('compare user preference and news')
        sim_mat = cosine_similarity(news_mat, topic_mat)

        np.save(data_path + 'similarity_matrix.npy', sim_mat)

        with open(data_path + 'id2topic.json', 'w') as f:
            json.dump(self.id2topic, f)

    def get_topic_matrix(self):
        """
        topic_words = [w1, w2, w3....]

        topic_vector.shape = (1, embedding_dim)

        user_matrix.shape = (num_topic, embedding_dim)

        """
        print('get topic matrix')

        topic_words_dict = self.config['topic_words']

        topic_matrix = np.empty((0, self.wordvec.embedding_dim))

        topic_id = 0
        for topic in topic_words_dict.keys():
            topic_words = topic_words_dict[topic]
            topic_vector = self.wordvec.avg_words_vector(topic_words)

            topic_matrix = np.append(topic_matrix, topic_vector, axis=0)

            self.id2topic[str(topic_id)] = topic
            topic_id += 1

        return topic_matrix

    def __gen_news_matrix(self):
        print('gen news matrix')

        with open(data_path + 'id2news.json', 'r') as f:
            id2news = json.load(f)

        news2vec = News2Vec(self.config)

        n_news = len(id2news.keys())

        news_matrix = np.empty((0, self.wordvec.embedding_dim))

        for i in range(n_news):
            news = id2news[str(i)]
            vec = news2vec.trans_one(news)
            news_matrix = np.append(news_matrix, vec, axis=0)

        np.save(data_path + 'news_matrix.npy', news_matrix)

    def __get_news_matrix(self):
        news_matrix_path = data_path + 'news_matrix.npy'

        if os.path.isfile(news_matrix_path):
            news_matrix = np.load(news_matrix_path)

        else:
            self.__gen_news_matrix()
            news_matrix = np.load(news_matrix_path)

        return news_matrix

