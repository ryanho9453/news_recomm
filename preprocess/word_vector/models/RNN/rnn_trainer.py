import tensorflow as tf
import numpy as np
import json
import os
import time
import argparse
import sys
from datetime import datetime

script_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = script_path.replace('models/RNN/', '')
preprocess_path = package_path + 'preprocess/'

sys.path.append(script_path)

from rnn_model import RNNModel
from rnn_reader import DataReader


def train_rnn(config, restore):
    n_gram = config['rnn_model']["n_gram"]
    batch_size = config['rnn_model']["batch_size"]

    data_reader = DataReader(config)

    num_sentence = data_reader.num_sentence

    config['rnn_model']['vocab_size'] = data_reader.vocab_size
    vocab_size = data_reader.vocab_size

    with tf.name_scope('data_in'):
        wordid_inputs = tf.placeholder(tf.int32, [batch_size, n_gram], name='data_in')

    with tf.name_scope('label_in'):
        target_inputs = tf.placeholder(tf.int32, [None, vocab_size], name='label_in')

    with tf.Session() as sess:
        model = RNNModel(config)

        model_save_path = script_path + 'saved_model/'
        ckpt_path = model_save_path + 'model.ckpt'
        meta_path = model_save_path + 'model.ckpt.meta'

        if not restore:
            train_op, loss, logits = model.build(wordid_inputs, target_inputs)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

        else:
            print('Restore saved model which stored at %s' % meta_path)
            train_op, loss, logits = model.build(wordid_inputs, target_inputs)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

        loss_summary = tf.summary.scalar('loss', loss)
        writer = tf.summary.FileWriter(script_path + 'tensor_log/', sess.graph)

        start_time = datetime.now()
        print('Training start at %s' % (start_time))
        for step in range(1, config['rnn_model']['steps'] + 1):

            # read_start_time = datetime.now()
            context_b, target_b = data_reader.next_batch()
            # read_time = datetime.now()-read_start_time
            # print('read_time: %s' % (read_time))

            feed_dict = {wordid_inputs: context_b,
                         target_inputs: target_b}

            sess.run(train_op, feed_dict=feed_dict)

            if step % 500 == 0:
                train_loss, loss_summ, prediction = sess.run(
                    [loss, loss_summary, logits], feed_dict=feed_dict)

                writer.add_summary(loss_summ, step)

                saver.save(sess, ckpt_path, global_step=tf.train.get_global_step())

                epoch_no = ((step * batch_size) // num_sentence) + 1

                print('')
                print('Save model at step = %s, epoch = %s' % (step, epoch_no))

                spend_time = datetime.now() - start_time
                print('loss = %s, step = %s (spend %s)'
                      % (train_loss, step, spend_time))

        saver.save(sess, ckpt_path)
        spend_time = datetime.now() - start_time
        print('Save model at final step = %s, spend %s' % (step, spend_time))

        graph = tf.get_default_graph()
        wordvec_tensor = graph.get_tensor_by_name('wordvec/embedding:0')
        wordvec = sess.run(wordvec_tensor)

        np.save(script_path + 'saved_model/word_vectors.npy', wordvec)

        with open(script_path + 'saved_model/word_id_converter.json', 'w') as f:
            json.dump(data_reader.word_id_converter, f)

