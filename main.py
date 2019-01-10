import os
import sys
import numpy as np
import json
from pprint import pprint
import argparse

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
data_path = package_path + 'data/'
utils_path = package_path + 'utils/'
preprocess_path = package_path + 'preprocess/'
wordvec_path = preprocess_path + 'word_vector/'

sys.path.append(utils_path)
sys.path.append(wordvec_path)
sys.path.append(preprocess_path)

from news_recomm import NewsRecommendor
from topic_news_matrix import TopicNewsMatrix
from mat import *


"""
news2vec.py -- convert docs to vectors

topic_news_matrix.py -- create topic doc similarity matrix


"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='news recommendation')
    parser.add_argument('--topic-name', dest='topic', default='topic_name', help='input your topic name in config')
    parser.add_argument('--reset', dest='reset', action='store_true', help='true if topic words change')
    parser.set_defaults(reset=False)
    args = parser.parse_args()

    with open(package_path + 'config.json', 'r', encoding='utf8') as f:
        config = json.load(f)

    if args.reset:
        topic_news_mat = TopicNewsMatrix(config)
        topic_news_mat.create()

    recomm = NewsRecommendor()
    result = recomm.get_top_news(args.topic)

    result_id = 1
    for news in result:
        print('<News %s>' % result_id)
        print(news)
        result_id += 1
