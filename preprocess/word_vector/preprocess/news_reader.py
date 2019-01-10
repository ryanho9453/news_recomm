import json
import os
import random

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'


class NewsReader:
    def __init__(self):
        self.path = data_path + 'news_data.json'

        self.max_n_news = 23000

    def pull_n_docs(self, train_size, test_size):
        with open(self.path, 'r') as f:
            news_list = json.load(f)

        if train_size > self.max_n_news:
            train_size = 23000

        elif test_size > self.max_n_news:
            test_size = 23000

        random.shuffle(news_list)

        train_news = news_list[0: train_size]
        test_news = news_list[train_size: train_size+test_size]

        return train_news, test_news

