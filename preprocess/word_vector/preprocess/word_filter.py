import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('preprocess/', '')

sys.path.append(script_path)


class WordFilter:
    def __init__(self, config):
        marks = ['（', '〔', '［', '｛', '《', '【', '〖', '〈', '(', '[' '{', '<',
                 '）', '〕', '］', '｝', '》', '】', '〗', '〉', ')', ']', '}', '>',
                 '“', '‘', '『', '』', '。', '？', '?', '！', '!', '，', ',', '', '；',
                 ';', '、', '：', ':', '……', '…', '——', '—', '－－', '－', '-', ' ',
                 '「', '」', '／', '/', ',', '.', '=', '+', '#', '\xa0', '\r\n', '@', '...', '\t', '\n',
                 '|', '%', '#', 'of', 'in', 'rss']

        with open(os.path.join(script_path, 'stopwords/stopword_cht.txt'), 'r') as a:
            stopwords_cht = [word.strip('\n') for word in a]

        with open(os.path.join(script_path, 'stopwords/terrier-stop.txt'), 'r') as a:
            stopwords_eng = [word.strip('\n') for word in a]

        self.stopwords = marks + stopwords_cht + stopwords_eng

        with open(os.path.join(script_path, 'speech_verb/speech_verb.txt'), 'r') as a:
            speech_verb = [word.strip('\n') for word in a]

        self.speech_verb = speech_verb

        if config['preprocess']['keep_speech_verb']:
            self.wf_list = self.__keep_speech_verb()

        else:
            self.wf_list = self.stopwords

    def get_wf_list(self):
        return self.wf_list

    def __keep_speech_verb(self):
        return list(set(self.stopwords) - set(self.speech_verb))
