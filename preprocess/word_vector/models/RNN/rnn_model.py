import tensorflow as tf


class RNNModel:
    def __init__(self, config):
        self.config = config

        self.learning_rate = config['rnn_model']['learning_rate']
        self.num_units = config['rnn_model']['num_units']
        self.embedding_dim = config['rnn_model']['embedding_dim']
        self.batch_size = config['rnn_model']['batch_size']

        self.vocab_size = config['rnn_model']['vocab_size']

    def build(self, data_in, label_in):
        pred, logits = self.__build_model(data_in)
        train_op, loss = self.__build_train_op(logits, label_in)

        return train_op, loss, logits

    def __build_model(self, data_in):

        cell = self.__get_rnn_layer(name='rnn1', num_units=self.num_units)

        wordvec_inputs = self.__get_wordvec_variable(data_in)

        initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # cell = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell=cell, inputs=wordvec_inputs, initial_state=initial_state)

        last = outputs[:, -1, :]
        logits = tf.layers.dense(inputs=last, units=self.vocab_size)
        pred = tf.nn.softmax(logits)

        return pred, logits

    def __build_train_op(self, logits, label_in):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_in)
            loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss)

        return train_op, loss

    def __get_rnn_layer(self, name, num_units, cell_type='basic'):
        with tf.variable_scope(name):
            if cell_type == 'basic':
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_units, activation=tf.nn.relu)

            elif cell_type == 'LSTM':
                cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)

        return cell

    def __get_wordvec_variable(self, wordid_inputs):
        with tf.variable_scope('wordvec', initializer=tf.contrib.layers.xavier_initializer()):
            wordvec = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim], dtype=tf.float32)

            # wordvec_inputs.shape = (batch_size, n_gram, embedding_dim)
            wordvec_inputs = tf.nn.embedding_lookup(wordvec, wordid_inputs, name='wordvec_inputs')

        return wordvec_inputs
