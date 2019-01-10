import os
import sys
import numpy as np
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')
data_path = package_path + 'data/'

sys.path.append(script_path)

from doclist_generator import DoclistGenerator
from my_count_vectorizer import MyCountVectorizer


class TermDocumentMatrix:

    def __init__(self, config):
        self.config = config
        self.max_df = config['preprocess']['max_df']
        self.min_df = config['preprocess']['min_df']

        self.n_docs = None
        self.n_words = None
        self.n_clean_words = None

        self.whitelist = self._read_whitelist(os.path.join(script_path, 'whitelist/whitelist.txt'))

        self.word_id_converter = dict()

    def create(self, save_matrix=False):
        dg = DoclistGenerator(self.config)
        train_doclist, test_doclist = dg.gen_n_docs(format='english')
        print('create term_document matrix')

        count_vect = MyCountVectorizer(max_df=self.max_df, min_df=self.min_df,
                                       config=self.config, dtype=np.uint8, whitelist=True)

        # <class 'scipy.sparse.csr.csr_matrix'>
        td_matrix_csr = count_vect.fit_transform(train_doclist)

        # shape(docs, words)
        td_matrix = td_matrix_csr.toarray()

        del td_matrix_csr

        self.n_docs = td_matrix.shape[0]
        self.n_words = td_matrix.shape[1]
        print('td_matrix shape : '+str(td_matrix.shape))

        word2id = count_vect.vocabulary_
        self.word_id_converter = self._gen_word_id_converter(word2id)

        vocab_size, corpus_size = self._collect_corpus_info(td_matrix)
        # self._calculate_p_w(td_matrix, corpus_size)

        if save_matrix is True:
            print('save td_matrix')
            np.save(data_path + 'term_document_matrix/td_matrix.npy', td_matrix)

        return td_matrix

    def load(self):
        print('load td_matrix')
        td_matrix = np.load(data_path + 'term_document_matrix/td_matrix.npy')
        print('td_matrix shape : ' + str(td_matrix.shape))

        return td_matrix

    def load_word_id_converter(self):
        with open(data_path + 'term_document_matrix/word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        return word_id_converter

    def _collect_corpus_info(self, td_matrix):
        vocab_size = td_matrix.shape[1]
        corpus_size = td_matrix.sum()

        # dict key is wordid
        df_dict = self.__gen_df_dict(td_matrix, vocab_size)
        tf_dict = self.__gen_tf_dict(td_matrix)

        self.corpus_info = dict()
        self.corpus_info['config'] = self.config
        self.corpus_info['vocab_size'] = int(vocab_size)
        self.corpus_info['corpus_size'] = int(corpus_size)
        self.corpus_info['df_dict'] = df_dict
        self.corpus_info['tf_dict'] = tf_dict

        with open(data_path + 'term_document_matrix/corpus_info.json', 'w', encoding='utf8') as f:
            json.dump(self.corpus_info, f)

        return vocab_size, corpus_size

    def __gen_df_dict(self, td_matrix, vocab_size):
        id2word = self.word_id_converter['id2word']

        df_dict = {}
        for w in range(vocab_size):
            df = np.count_nonzero(td_matrix[:, w])
            df_dict[id2word[str(w)]] = int(df) / self.n_docs

        return df_dict

    def __gen_tf_dict(self, td_matrix):
        id2word = self.word_id_converter['id2word']

        tf_dict = {}
        tf_array = td_matrix.sum(axis=0)
        for i, tf in enumerate(tf_array):
            tf_dict[id2word[str(i)]] = int(tf)

        return tf_dict

    def _calculate_p_w(self, td_matrix, corpus_size):
        tf_array = td_matrix.sum(axis=0)
        p_w = tf_array / corpus_size  # p_w.shape (1, w)

        np.save(data_path + 'p_w.npy', p_w)

    def _gen_word_id_converter(self, word2id):
        word_id_converter = {}

        # word2id's id was numpy.int64, not json serializable
        new_word2id = {k: int(v) for k, v in word2id.items()}

        id2word = {str(v): k for k, v in new_word2id.items()}

        word_id_converter['word2id'] = new_word2id
        word_id_converter['id2word'] = id2word

        with open(data_path + 'term_document_matrix/word_id_converter.json', 'w', encoding='utf8') as f:
            json.dump(word_id_converter, f)

        with open(data_path + 'word_id_converter.json', 'w', encoding='utf8') as f:
            json.dump(word_id_converter, f)

        return word_id_converter

    def _read_whitelist(self, whitelist_path):
        with open(whitelist_path, 'r') as f:
            lines = f.readlines()

        termlist = list()
        for line in lines:
            newline = line.replace('\n', '')
            termlist.append(newline)

        return termlist
