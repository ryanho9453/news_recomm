import os
import sys
import numpy as np
import json
from collections import Counter

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'

sys.path.append(script_path)

from doclist_generator import DoclistGenerator
from my_count_vectorizer import MyCountVectorizer


class NGramData:
    def __init__(self, config):
        self.config = config
        self.max_df = config['preprocess']['max_df']
        self.min_df = config['preprocess']['min_df']

        self.batch_size = config['rnn_model']['batch_size']
        self.n_gram = config['rnn_model']['n_gram']
        self.stride = config['rnn_model']['stride']
        self.random_pull = config['rnn_model']['random_pull']
        self.use_word_filter = config['rnn_model']['use_word_filter']

        self.num_sentence = None
        self.vocab_size = None

        self.whitelist = self._read_whitelist(os.path.join(script_path, 'whitelist/whitelist.txt'))

        self.dg = DoclistGenerator(config)

        self.word_id_converter = dict()

    def create(self, save=False):
        """
        use MyCountVectorizer to filter max_df, min_df words, and create vocabulary

        then use the vocab to turn the termlist to idlist (therefore, max_df words would be filter out)

        """
        train_doclist, test_doclist = self.dg.gen_n_docs(format='termlist')

        word_id_converter = self._gen_vocabulary()

        train_doclist = self._termlist_to_idlist(train_doclist, word2id=word_id_converter['word2id'])

        context = []
        target = []

        for doc in train_doclist:
            doc_len = len(doc)
            # read through the doc
            for i in range(0, doc_len - self.n_gram, self.stride):
                # x -- context in n-gram
                context.append(doc[i:(i + self.n_gram)])
                # y -- target in n-gram
                target.append(doc[i + self.n_gram])

        context = np.array(context)
        target = np.array(target)

        self.num_sentence = len(context)

        print('%s sentences (n_gram: %s, stride: %s)' % (len(context), self.n_gram, self.stride))

        # batch_size = number of x in a batch
        # num_batch = number of batch in dataset
        num_batch = len(context) // self.batch_size
        """
        one_batch x =  [[ 11 (wordid), 36, 77, ......],   (one sentence)
                       [ 45, 34, 76, ......],
                         .......
                        .....................]] 
        
        one_batch y =  [ 23 (one sentence), 65, 77, 19, ...... ] 

    
        one_batch.shape = (batch_size, n_gram)
                           batch_size = num_sentence in one batch
                           
        x = wordid_inputs
        x.shape = (num_batch, batch_size, n_gram)
        
        # every batch will be full

        """

        x = np.reshape(context[:(num_batch * self.batch_size)], [num_batch, self.batch_size, -1])

        y = np.reshape(target[:(num_batch * self.batch_size)], [num_batch, self.batch_size])

        with open(package_path + 'models/RNN/saved_model/word_id_converter.json', 'w') as f:
            json.dump(self.word_id_converter, f)

        if save:
            np.save(data_path + 'n_gram_data/x.npy', x)
            np.save(data_path + 'n_gram_data/y.npy', y)

        return x, y

    def _termlist_to_idlist(self, doclist, word2id):
        """
        doclist = [['瑞凡', '聞', '家明', '的', '味道'],
                   ['覺得', '喜翻', ......           ]]

        new_doclist = [[12, 57, 45,  79, 53],
                       [55, .......        ]]

        """

        new_doclist = []
        for termlist in doclist:
            idlist = [word2id[term] for term in termlist if term in word2id]
            new_doclist.append(idlist)

        return new_doclist

    def _gen_vocabulary(self):
        """
        use CountVectorizer to build vocabulary, because we need to calculate df and filter words with max_df, min_df

        """
        train_doclist, test_doclist = self.dg.gen_n_docs(format='english')

        count_vect = MyCountVectorizer(max_df=self.max_df, min_df=self.min_df,
                                       config=self.config, dtype=np.uint8, whitelist=True)

        # <class 'scipy.sparse.csr.csr_matrix'>
        td_matrix_csr = count_vect.fit_transform(train_doclist)

        del td_matrix_csr

        word2id = count_vect.vocabulary_
        word_id_converter = self._gen_word_id_converter(word2id)

        self.vocab_size = len(word_id_converter['word2id'].keys())

        print('Vocabulary Done : %s words (max_df: %s , min_df: %s)' % (self.vocab_size, self.max_df, self.min_df))

        return word_id_converter

    def _gen_word_id_converter(self, word2id):
        word_id_converter = {}

        # word2id's id was numpy.int64, not json serializable
        new_word2id = {k: int(v) for k, v in word2id.items()}

        id2word = {str(v): k for k, v in new_word2id.items()}

        word_id_converter['word2id'] = new_word2id
        word_id_converter['id2word'] = id2word

        self.word_id_converter = word_id_converter

        return word_id_converter

    def _read_whitelist(self, whitelist_path):
        with open(whitelist_path, 'r') as f:
            lines = f.readlines()

        termlist = list()
        for line in lines:
            newline = line.replace('\n', '')
            termlist.append(newline)

        return termlist