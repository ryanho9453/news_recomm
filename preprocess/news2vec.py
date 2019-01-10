import os
import sys
import numpy as np
import json
import jieba

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')

wordvec_path = script_path + 'word_vector/'

utils_path = package_path + 'utils/'
sys.path.append(utils_path)
sys.path.append(wordvec_path)

from mat import *
from word_vector import WordVector


class News2Vec:
    def __init__(self, config):
        self.config = config

        self.wordvec = WordVector(model=config['word_vector']['model'])
        self.wordvec.restore()

        self.embedding_dim = self.wordvec.word_vectors.shape[1]

    def trans_many(self, docs):
        """
        input docs = list of doc

        """
        doc_vec_collect = np.empty((0, self.embedding_dim))
        for doc in docs:
            doc_vec = self.trans_one(doc)
            np.append(doc_vec_collect, doc_vec, axis=0)

        return doc_vec_collect

    def trans_one(self, doc):
        """
        return (1, embedding_dim) array of a doc

        """

        doc_vec = np.empty((0, self.embedding_dim))

        seglist = jieba.cut(doc)
        for term in seglist:
            if term in self.wordvec.word2id.keys():
                wv = self.wordvec.get_word_vector(term)
                doc_vec = np.append(doc_vec, wv, axis=0)

        if doc_vec.shape[0] == 0:
            doc_vec = np.zeros((1, self.embedding_dim))

        else:
            doc_vec = np.mean(doc_vec, axis=0)[np.newaxis, :]

        return doc_vec

