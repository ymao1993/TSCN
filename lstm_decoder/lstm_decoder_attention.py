"""
The code implements LSTM decoder module used for caption generation.

The code is modified from the tensorflow implementation fo show-attend-and-tell source code
https://github.com/yunjey/show-attend-and-tell.
"""

# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is number of feature vectors extracted from a lower conv-layer.
# D is dimension of image feature vector extracted from a lower conv-layer.
# T is the maximum number of time steps of LSTM, which is equal to the maximum length of captions.
# V is vocabulary size (about 10000).
# H is dimension of hidden state (default is 1024).
# M is dimension of word vector which is embedding size (default is 512).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        """ Predict the initial cell state and hidden state from the average of the annotation vectors.

        Note: By definition, cell state and hidden state should encode the previous hidden state,
        the previous word embedding and the feature vectors. At the first time step, we don't have the
        previous hidden state, the previous word embedding so the best thing we can do is to initialize
        cell state and hidden state from the feature vectors.
        """
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        """ Convert input words to word embeddings of size M.
        """
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        """ Convert features into feature embedding (by changing basis).
        """
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        """ Compute context vector from features and hidden state.
        """
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            # Compute
            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)  # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])  # (N, L)

            # Compute the blending weights.
            # (alpha shape: [batch_size, L])
            alpha = tf.nn.softmax(out_att)

            # Compute the context vector by soft-attention mechanism: simply weighted-combining the feature vectors.
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        """ Scale the attention by a scalar computed from the hidden state.
        """
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.mul(beta, context, name='selected_context')  # (N, L)
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        """ Decode the LSTM output into logits for the next character.
        """
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + 'batch_norm'))

    def build_model(self):
        """
        Build the model in training mode.
        """
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        # Construct input word sequences and output word sequences.
        # (caption_in shape: [batch_size, H])
        # (caption_out shape: [batch_size, H])
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))

        # Batch-normalize feature vectors.
        features = self._batch_norm(features, mode='train', name='conv_features')

        # Initialize LSTM states from the feature vectors.
        # cell_state shape: [batch_size, H]
        # hidden_state shape: [batch_size, H]
        cell_state, hidden_state = self._get_initial_lstm(features=features)

        # Convert input word sequences to input word embeddings.
        # x shape: [batch_size, T, M]
        x = self._word_embedding(inputs=captions_in)

        # Project features by changing the basis.
        # features_proj: [batch_size, D]
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        for t in range(self.T):
            # (context shape: [N, D])
            # (alpha shape: [N, L])
            context, alpha = self._attention_layer(features, features_proj, hidden_state, reuse=(t != 0))
            alpha_list.append(alpha)

            # Scale the context vector.
            if self.selector:
                context, beta = self._selector(context, hidden_state, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (cell_state, hidden_state) = lstm_cell(inputs=tf.concat(1, [x[:, t, :], context]),
                                                          state=[cell_state, hidden_state])

            logits = self._decode_lstm(x[:, t, :], hidden_state, context, dropout=self.dropout, reuse=(t != 0))
            loss += tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, captions_out[:, t]) * mask[:, t])

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((1. - alphas_all) ** 2)
            loss += alpha_reg
        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        """
        Build the model in test mode using argmax method for sampling.
        """
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        cell_state, hidden_state = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                batch_size = tf.shape(features)[0]
                x = self._word_embedding(inputs=tf.fill([batch_size], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, hidden_state, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, hidden_state, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (cell_state, hidden_state) = lstm_cell(inputs=tf.concat(1, [x, context]),
                                                          state=[cell_state, hidden_state])

            logits = self._decode_lstm(x, hidden_state, context, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))  # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        sampled_captions = tf.transpose(tf.pack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_captions
