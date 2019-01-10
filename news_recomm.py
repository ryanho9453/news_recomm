import os
import sys
import numpy as np
import json

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
wordvec_path = package_path + 'preprocess/word_vector/'
data_path = package_path + 'data/'
utils_path = package_path + 'utils/'

sys.path.append(utils_path)
sys.path.append(wordvec_path)
sys.path.append(package_path + 'preprocess/')

from mat import *


class NewsRecommendor:
    def __init__(self):
        """
        get top n news on sim_matrix

        sim_matrix axis0 - news_id
                   axis1 - topic_id

        """
        self.sim_matrix = np.load(data_path + 'similarity_matrix.npy')

        with open(data_path + 'id2topic.json', 'r') as f:
            self.id2topic = json.load(f)

        self.topic2id = {v: int(k) for k, v in self.id2topic.items()}

        with open(data_path + 'id2news.json', 'r') as f:
            self.id2news = json.load(f)

    def get_top_news(self, topic, n_result=10):
        """
        top n similar news

        """

        # id type string
        for id in self.id2topic.keys():
            if self.id2topic[id] == topic:
                # news_ids = list of news id
                top_news_ids = np.argpartition(self.sim_matrix[:, int(id)], -1 * n_result)[-1 * n_result:][::-1]

        news_result = []
        for news_id in top_news_ids:
            news = self.id2news[str(news_id)]
            news_result.append(news)

        return news_result

    def get_top_news_with_sim(self, topic, n_result):
        # id type string
        topic_id = self.topic2id[topic]

        # news_ids = list of news id
        top_news_ids = np.argpartition(self.sim_matrix[:, topic_id], -1 * n_result)[-1 * n_result:][::-1]

        news_result = []
        for news_id in top_news_ids:
            sim = self.sim_matrix[news_id, topic_id]
            news = self.id2news[str(news_id)]
            news_result.append((sim, news))

        return news_result
