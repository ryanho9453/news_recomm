import json
from pprint import pprint
import sys
import os

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/prior_cluster/', '')

sys.path.append(script_path)


class ClusterEditor:

    def __init__(self):

        self.prior_cluster_path = script_path + 'prior_cluster.json'

    def get_all_prior_word(self):
        with open(self.prior_cluster_path, 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        wordlist = list(prior_cluster.keys())

        return wordlist

    def get_raw_cluster(self):
        with open(self.prior_cluster_path, 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        return prior_cluster

    def get_cluster(self, cluster_no=None):
        with open(self.prior_cluster_path, 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        clusters = dict()
        for word, cluster in prior_cluster.items():
            if 'in' in cluster.keys():
                in_cluster = cluster['in'][0]
                if in_cluster not in clusters.keys():
                    clusters[in_cluster] = [word]

                else:
                    clusters[in_cluster].append(word)

        if cluster_no is not None:
            clusters = clusters[cluster_no]

        return clusters

    def add_word(self, word_in_cluster=None, word_not_in_cluster=None):
        """
        word_in_cluster = {old_word: new_word}
        ===> new_word is in the cluster of old_word

        word_not_in_cluster = {

        """

        with open(self.prior_cluster_path, 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        if word_in_cluster is not None:
            for oldword, new_word in word_in_cluster.items():
                cluster = prior_cluster[oldword]['in'][0]
                for word in new_word:
                    if word in prior_cluster.keys():
                        if 'in' in prior_cluster[word].keys():
                            old_cluster = prior_cluster[word]['in'][0]
                            print('add in : '+word+' already in cluster '+str(old_cluster))

                        elif 'not_in' in prior_cluster[word].keys():
                            prior_cluster[word]['in'] = [cluster]

                    else:
                        prior_cluster[word] = {'in': [cluster]}

        if word_not_in_cluster is not None:
            for oldword, new_word in word_not_in_cluster.items():
                cluster = prior_cluster[oldword]['in'][0]
                for word in new_word:
                    if word in prior_cluster.keys():
                        if 'in' in prior_cluster[word].keys():
                            old_cluster = prior_cluster[word]['in'][0]
                            print('add not in : '+word + ' already in cluster ' + str(old_cluster))

                        elif 'not_in' in prior_cluster[word].keys():
                            if cluster not in prior_cluster[word]['not_in']:
                                prior_cluster[word]['not_in'].append(cluster)

                            else:
                                print('add not in : ' + word + ' already not in cluster ' + str(cluster))

                    else:
                        prior_cluster[word] = {'not_in': [cluster]}

        with open(self.prior_cluster_path, 'w', encoding='utf8') as f:
            json.dump(prior_cluster, f, ensure_ascii=False)

    def find_similar_word_for_cluster(self, cluster_no, n_newword=15):
        clusters = self.get_cluster()

        keyword = clusters[cluster_no]

        word_predictor = WordPredictor(config, term_similarity_matrix=True)
        result_dict = word_predictor.find_similar_word(keyword, n_target=n_newword)

        return result_dict

    def find_new_word_for_cluster(self, cluster_no, n_newword=15):

        wordlist = self.get_all_prior_word()  # "in" word and "not_in" word would be different

        clusters = self.get_cluster()

        keyword = clusters[cluster_no]

        word_predictor = WordPredictor(config, term_similarity_matrix=True)
        result_dict = word_predictor.find_similar_word(keyword, n_target=n_newword)

        new_dict = dict()
        for key, value in result_dict.items():
            new_list = list()
            for predicted_word in value:
                if predicted_word not in wordlist:
                    new_list.append(predicted_word)

            new_dict[key] = new_list

        return new_dict

    def check_cluster_conflict(self):
        with open(self.prior_cluster_path, 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        for word, cluster in prior_cluster.items():
            if 'in' in cluster.keys() and 'not_in' in cluster.keys():
                print(word+' with cluster conflict')

            if 'in' in cluster.keys():
                if len(cluster['in']) > 1:
                    print(word+' with multiple cluster')

            if 'not_in' in cluster.keys():
                k = list(set(cluster['not_in']))
                prior_cluster[word]['not_in'] = k

        with open(self.prior_cluster_path, 'w', encoding='utf8') as f:
            json.dump(prior_cluster, f, ensure_ascii=False)

