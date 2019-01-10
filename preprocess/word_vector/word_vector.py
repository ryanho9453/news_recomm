import numpy as np
import json
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
sys.path.append(package_path)
sys.path.append(package_path + 'preprocess/')
sys.path.append(package_path + 'models/RNN/')
sys.path.append(package_path + 'models/LDA/')

from rnn_trainer import train_rnn
from lda_trainer import train_lda


class WordVector:
    def __init__(self, model):
        self.model = model

        self.word_vectors = None
        self.word2id = None
        self.id2word = None

        self.vocab_size = None
        self.embedding_dim = None

    def train(self, config):
        if self.model == 'LDA':
            train_lda(config)

        elif self.model == 'RNN':
            train_rnn(config, restore=False)

    def restore(self):
        if self.model == 'LDA':
            self.word_vectors = np.load(package_path + 'models/LDA/saved_model/word_vectors.npy')
            with open(package_path + 'models/LDA/saved_model/word_id_converter.json', 'r') as f:
                word_id_converter = json.load(f)

            self.word2id = word_id_converter['word2id']
            self.id2word = word_id_converter['id2word']

            self.vocab_size = self.word_vectors.shape[0]
            self.embedding_dim = self.word_vectors.shape[1]

        elif self.model == 'RNN':
            self.word_vectors = np.load(package_path + 'models/RNN/saved_model/word_vectors.npy')
            with open(package_path + 'models/RNN/saved_model/word_id_converter.json', 'r') as f:
                word_id_converter = json.load(f)

            self.word2id = word_id_converter['word2id']
            self.id2word = word_id_converter['id2word']

            self.vocab_size = self.word_vectors.shape[0]
            self.embedding_dim = self.word_vectors.shape[1]

        print('model restored')

    def get_word_vector(self, word):
        if word in self.word2id.keys():
            wordid = self.word2id[word]

            return self.word_vectors[wordid, :][np.newaxis, :]

        else:
            pass

    def get_words_vector(self, words):
        wvs = np.empty((0, self.embedding_dim))

        for word in words:
            if word in self.word2id.keys():
                wordid = self.word2id[word]
                wv = self.word_vectors[wordid, :][np.newaxis, :]
                wvs = np.append(wvs, wv, axis=0)

        return wvs

    def avg_words_vector(self, words):

        wvs = self.get_words_vector(words)

        avg = np.mean(wvs, axis=0)[np.newaxis, :]

        return avg

    def find_similar_word(self, word, n_target=10):
        if word in self.word2id.keys():
            # word_sim_mat.shape = (1, vocab_size)
            word_sim_mat = self.__gen_word_sims(word)

            n_target_plus_itself = n_target + 1
            top_wordid = np.argsort(word_sim_mat[0, :])[- 1 * n_target_plus_itself:][::-1].tolist()
            print(top_wordid)
            top_wordid.remove(self.word2id[word])

            result = []
            for wordid in top_wordid:
                result.append(self.id2word[str(wordid)])

            return result

        else:
            print('word %s not in dictionary' % word)

    def __gen_word_sims(self, word):
        if self.model == 'LDA':
            self.word_vectors = np.load(package_path + 'models/LDA/saved_model/word_vectors.npy')

            wordid = self.word2id[word]
            wv = self.word_vectors[wordid, :][np.newaxis, :]

            word_sim_matrix = cosine_similarity(wv, self.word_vectors)

            return word_sim_matrix

        elif self.model == 'RNN':
            self.word_vectors = np.load(package_path + 'models/RNN/saved_model/word_vectors.npy')

            wordid = self.word2id[word]
            wv = self.word_vectors[wordid, :][np.newaxis, :]

            word_sim_matrix = cosine_similarity(wv, self.word_vectors)

            return word_sim_matrix

        print('word sim matrix done')
