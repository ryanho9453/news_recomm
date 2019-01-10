import jieba
import re
import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')

sys.path.append(script_path)

from news_reader import NewsReader
from word_filter import WordFilter

whitelist_path = os.path.join(script_path, 'whitelist/whitelist.txt')
jieba.load_userdict(whitelist_path)


class DoclistGenerator:
    def __init__(self, config):
        train_size = config['preprocess']['train_size']
        test_size = config['preprocess']['test_size']
        use_word_filter = config['preprocess']['use_word_filter']

        if use_word_filter:
            wf = WordFilter(config)
            self.wf_list = wf.get_wf_list()
            print('use word filter')

        else:
            self.wf_list = []

        news_reader = NewsReader()
        self.train_news, self.test_news = news_reader.pull_n_docs(train_size, test_size)

    def gen_n_docs(self, format=None, by_sentence=False):

        if format == 'english':
            train_doclist = self._english_format(self.train_news, by_sentence)
            test_doclist = self._english_format(self.test_news, by_sentence)

        elif format == 'termlist':
            train_doclist = self._termlist_format(self.train_news, by_sentence)
            test_doclist = self._termlist_format(self.test_news, by_sentence)

        else:
            print('specify your doclist format')

        return train_doclist, test_doclist

    def _english_format(self, docs, by_sentence):
        """
        :param docs: list of doc (doc is type dict)
        :return: list of doc (do  is type string)

        Countvectorizer 僅用於 english format 文章
        為了用 Countvectorizer 將 docs 轉成 term_document matrix
        將 '我要成為電腦工程師' 轉成 string '我 要 成為 電腦 工程師' , 在收集成 list

        """
        whitelist = self._read_whitelist(whitelist_path)

        if by_sentence:
            sentence_list = []
            for news_dict in docs:
                seglist = jieba.cut(news_dict)
                sentence = ''
                for term in seglist:
                    term = term.lower()
                    # 有數字的term全部去除, 以whitelist的方式列入考量
                    if re.search('[0-9]', term):
                        if term in whitelist:
                            if sentence != '':
                                sentence += ' '
                                sentence += str(term)
                            else:
                                sentence += str(term)

                    else:
                        if term == '。':
                            sentence_list.append(sentence)
                            sentence = ''

                        elif term not in self.wf_list:
                            if sentence != '':
                                sentence += ' '
                                sentence += str(term)
                            else:
                                sentence += str(term)

            return sentence_list

        else:
            doclist = []
            for news_dict in docs:
                seglist = jieba.cut(news_dict)
                string = ''
                for term in seglist:
                    term = term.lower()
                    # 有數字的term全部去除, 以whitelist的方式列入考量
                    if re.search('[0-9]', term):
                        if term in whitelist:
                            if string != '':
                                string += ' '
                                string += str(term)
                            else:
                                string += str(term)

                    else:
                        if term not in self.wf_list:
                            if string != '':
                                string += ' '
                                string += str(term)
                            else:
                                string += str(term)

                doclist.append(string)

            return doclist

    def _termlist_format(self, docs, by_sentence):
        """

        將文章'我要成為電腦工程師' 轉成 list [我, 要, 成為, 電腦, 工程師]

        """
        whitelist = self._read_whitelist(whitelist_path)

        if by_sentence:
            sentence_list = []
            for news_dict in docs:
                seglist = jieba.cut(news_dict)
                sentence = []
                for term in seglist:
                    term = term.lower()
                    if re.search('[0-9]', term):
                        if term in whitelist:
                            sentence.append(term)
                    else:
                        if term == '。':
                            sentence_list.append(sentence)
                            sentence = []

                        elif term not in self.wf_list:
                            sentence.append(term)

            return sentence_list

        else:
            doclist = []
            for news_dict in docs:
                seglist = jieba.cut(news_dict)
                termlist = []
                for term in seglist:
                    term = term.lower()
                    # 有數字的term全部去除, 以whitelist的方式列入考量
                    if re.search('[0-9]', term):
                        if term in whitelist:
                            termlist.append(term)
                    else:
                        if term not in self.wf_list:
                            termlist.append(term)

                doclist.append(termlist)

            return doclist

    def _read_whitelist(self, whitelist_path):
        with open(whitelist_path, 'r') as f:
            lines = f.readlines()

        termlist = list()
        for line in lines:
            newline = line.replace('\n', '')
            termlist.append(newline)

        return termlist
