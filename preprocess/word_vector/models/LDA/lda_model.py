import sys
import os
import numpy as np
from scipy.special import gammaln
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/', '')
preprocess_path = package_path + 'preprocess/'
utils_path = package_path + 'utils/'


sys.path.append(script_path)
sys.path.append(preprocess_path)
sys.path.append(utils_path)

from timer import Timer


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """

    A = np.random.multinomial(1, p).argmax()

    return A


def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.

             w1      w2      w3
    doc1 [   0       6        3

    return [w2*6  , w3*3 .....

    """
    for idx in vec.nonzero()[0]:
        for i in range(int(vec[idx])):
            yield idx


def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


class LdaModel:

    def __init__(self, n_topics, use_prior_cluster, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

        if use_prior_cluster:
            self.prior_wordids, self.wordid2cluster = self._prepare_prior_cluster()

        else:
            self.prior_wordids = list()
            self.wordid2cluster = dict()

    def _prepare_prior_cluster(self):
        """
        prior_wordids = list of wordid in prior_cluster.json

        worid2cluster = {word_id: "in": [1, .... (clusters)]
                                  "not_in": [1, .... (clusters)]
                        }

        """

        with open(script_path + 'saved_model/word_id_converter.json', 'r', encoding='utf8') as f:
            word_id_converter = json.load(f)

        word2id = word_id_converter['word2id']

        with open(script_path + 'prior_cluster/prior_cluster.json', 'r', encoding='utf8') as f:
            prior_cluster = json.load(f)

        prior_wordids = list()
        wordid2cluster = dict()
        for word, cluster in prior_cluster.items():
            if word in word2id.keys():
                wordid = word2id[word]
                prior_wordids.append(wordid)
                wordid2cluster[wordid] = cluster

        return prior_wordids, wordid2cluster

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape
        self.nmz = np.zeros((n_docs, self.n_topics))
        self.nzw = np.zeros((self.n_topics, vocab_size))

        self.nm = np.zeros(n_docs)

        self.nz = np.zeros(self.n_topics)
        self.topics = {}

        for m in range(n_docs):
            for i, w in enumerate(word_indices(matrix[m, :])):
                z = np.random.randint(self.n_topics)
                self.nmz[m, z] += 1
                self.nm[m] += 1
                self.nzw[z, w] += 1
                self.nz[z] += 1
                self.topics[(m, i)] = z

    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).

        2 way to define the prior , in one cluster, not_in multiple clusters

        p_z_with_prior = [......., in=0.8 , ....]

        """

        if w in self.prior_wordids:
            if 'in' in self.wordid2cluster[w].keys():
                p_z_with_prior = np.zeros((1, self.n_topics))

                # what cluster is w in
                prior_cluster = self.wordid2cluster[w]['in'][0]

                p_z_with_prior[:, prior_cluster] = 0.8

                other_index = [i for i in range(self.n_topics)]
                other_index.remove(prior_cluster)

                p_z_with_prior[:, other_index] = 0.2 / len(other_index)

                p_z_with_prior /= np.sum(p_z_with_prior)

                p_z_with_prior = p_z_with_prior.tolist()[0]

                return p_z_with_prior

            elif 'not_in' in self.wordid2cluster[w].keys():
                """
                p_z_with_prior = [....result of gibbs sampling......]
                
                w not in 0, 1, 2
                
                p_z_with_prior = [0, 0, 0, ....result of gibbs sampling......]
                
                """
                vocab_size = self.nzw.shape[1]

                # the probability of w on topic z
                left = (self.nzw[:, w] + self.beta) / \
                       (self.nz + self.beta * vocab_size)

                # the probability of doc m on topic z
                right = (self.nmz[m, :] + self.alpha) / \
                        (self.nm[m] + self.alpha * self.n_topics)
                p_z_with_prior = left * right
                # normalize to obtain probabilities
                p_z_with_prior /= np.sum(p_z_with_prior)

                excluding_clusters = self.wordid2cluster[w]['not_in']   # list of excluding clusters

                for cluster in excluding_clusters:
                    p_z_with_prior[cluster] = 0

                p_z_with_prior /= np.sum(p_z_with_prior)

                return p_z_with_prior

            else:
                print('no in or not_in in word_cluster.json for the word')
                sys.exit(1)

        else:
            vocab_size = self.nzw.shape[1]
            left = (self.nzw[:, w] + self.beta) / \
                (self.nz + self.beta * vocab_size)
            right = (self.nmz[m, :] + self.alpha) / \
                (self.nm[m] + self.alpha * self.n_topics)
            p_z = left * right

            # normalize to obtain probabilities
            p_z /= np.sum(p_z)

            return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """

        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in range(self.n_topics):
            lik += log_multi_beta(self.nzw[z, :]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in range(n_docs):
            lik += log_multi_beta(self.nmz[m, :]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi_pzw(self):
        """
        Compute phi = p(w|z).
                pzw = p(z|w)
        """

        phi = self.nzw + self.beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]

        p_zw = self.nzw.copy()
        p_zw /= np.sum(p_zw, axis=0)[np.newaxis, :]

        # phi.shape (z, w)  pzw.shape(w, z)
        return [phi, p_zw.T]

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.

        matrix shape(#doc, #word)

        """
        n_docs, vocab_size = matrix.shape

        self._initialize(matrix)

        for it in range(maxiter):
            timer = Timer()
            timer.start()

            print('--- iter '+str(it))

            for m in range(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m, i)]
                    self.nmz[m, z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z, w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m, z] += 1
                    self.nm[m] += 1
                    self.nzw[z, w] += 1
                    self.nz[z] += 1
                    self.topics[(m, i)] = z

            timer.print_time()
            print('--- end iter')

            yield self.phi_pzw()
