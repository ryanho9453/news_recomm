import numpy as np
import sys
import os

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/RNN/', '')
preprocess_path = package_path + 'preprocess/'

sys.path.append(script_path)
sys.path.append(preprocess_path)

from n_gram_data import NGramData


class DataReader:
    def __init__(self, config):
        self.config = config
        self.batch_index = 0

        self.batch_size = config['rnn_model']['batch_size']

        ngram_data = NGramData(config)
        self.context, self.target = ngram_data.create(save=False)

        self.word_id_converter = ngram_data.word_id_converter

        self.num_sentence = ngram_data.num_sentence

        self.num_batch = self.context.shape[0]
        self.vocab_size = ngram_data.vocab_size

    def next_batch(self):
        """
        every batch will be full

        self.target.shape = (num_batch, batch_size)
        one batch y = [1, 2, 3,  4, ....]

        y_onehot = [[0, 1, 0, 0, ...],     (one_batch)
                    [0, 0, 1, 0, ...],
                    [0, 0, 0, 1, ....],
                    .....
                    ........         ]]

        """

        if self.batch_index >= self.num_batch:
            self.batch_index = 0

        x = self.context[self.batch_index]
        y = self.target[self.batch_index]

        y_onehot = np.zeros((self.batch_size, self.vocab_size), dtype=np.bool)

        for i in range(self.batch_size):
            wordid_yi = y[i]
            y_onehot[i][wordid_yi] = 1

        self.batch_index += 1

        return x, y_onehot
