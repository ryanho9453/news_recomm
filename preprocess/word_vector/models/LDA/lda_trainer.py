import os
import sys
import numpy as np
import json

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/LDA/', '')
preprocess_path = package_path + 'preprocess/'

sys.path.append(script_path)
sys.path.append(preprocess_path)

from lda_model import LdaModel
from term_document_matrix import TermDocumentMatrix


def train_lda(config):
    n_topics = config['lda_model']['n_topics']
    alpha = config['lda_model']['alpha']
    beta = config['lda_model']['beta']
    maxiter = config['lda_model']['maxiter']
    use_prior_cluster = config['lda_model']['use_prior_cluster']

    # initialize
    print('initialize LDA model')
    maxlike = -1 * 10 ** 100
    opt_iter = 0
    likelihood_in_iters = dict()

    matrix = TermDocumentMatrix(config)
    td_matrix = matrix.create(save_matrix=True)
    word_id_converter = matrix.word_id_converter

    lda_model = LdaModel(n_topics=n_topics, alpha=alpha, beta=beta,
                         use_prior_cluster=use_prior_cluster)

    for i, phi_pzw in enumerate(lda_model.run(matrix=td_matrix, maxiter=maxiter)):
        like = lda_model.loglikelihood()

        likelihood_in_iters[i] = like

        # update best maximum likelihood and optimal phi = p(w| z)
        if like > maxlike:
            print('update optimal')
            maxlike = like
            opt_iter = i
            # opt_phi = phi_pzw[0]
            opt_pzw = phi_pzw[1]

        else:
            print('no update')

    print('save lda model')
    model_info = dict()
    model_info['maximum_likelihood'] = maxlike  # int
    model_info['optimal_iteration'] = opt_iter
    model_info['likelihood_in_iters'] = likelihood_in_iters

    with open(script_path + 'saved_model/lda_model_info.json', 'w', encoding='utf8') as f:
        json.dump(model_info, f)

    with open(script_path + 'saved_model/word_id_converter.json', 'w', encoding='utf8') as f:
        json.dump(word_id_converter, f)

    np.save(script_path + 'saved_model/word_vectors.npy', opt_pzw)

