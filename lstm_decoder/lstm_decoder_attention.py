"""
The code implements LSTM decoder module used for caption generation.

The basic idea is the same with show-and-tell model except that we concatenate the the video feature embedding with the
input word embedding and feed them as input at every time steps. In the original show-and-tell model, the image feature
embedding is only fed as input at the first time step.

This version is built based on lstm_decoder_scratch.py. The main modification is adding the temporal attention
mechanism. A context vector is constructed by weightedly combining frame-level feature vectors (annotation vectors) of
the sampled frames. The blending weights are computed from the hidden state (which encodes the information about the
context) and the annotation vectors.

The code is modified from show-and-tell source code https://github.com/tensorflow/models/tree/master/research/im2txt.
"""

import numpy as np
import tensorflow as tf
import const_config


class LSTMDecoderAttention(object):
    """ LSTM Decoder module.
    """

    def __init__(self, config, mode):
        """Basic setup.

        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "inference".
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode

        # Use Xavier initializer
        self.initializer = tf.contrib.layers.xavier_initializer()

        # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
        self.seq_embeddings = None

        # A float32 Tensor with shape [batch_size, num_segments, input_feature_size].
        self.input_features = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.feature_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_losses = None

        # A float32 Tensor with shape [batch_size * padded_length].
        self.target_cross_entropy_loss_weights = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def build_feature_embeddings(self):
        """Builds the input sequence embeddings.
        Inputs:
          self.input_seqs
        Outputs:
          self.seq_embeddings
        """
        self.input_features = tf.placeholder(dtype=tf.float32,
                                             shape=[self.config.batch_size,
                                             self.config.num_segments,
                                             self.config.input_feature_size],
                                             name="input_features")
        reshaped_feature = tf.reshape(self.input_features, (self.config.batch_size, -1))
        intermediate_layer = tf.contrib.layers.fully_connected(
            inputs=reshaped_feature,
            num_outputs=self.config.num_units_intermediate_fc,
            activation_fn=tf.nn.tanh,
            weights_initializer=self.initializer,
            biases_initializer=None,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
            biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength))
        with tf.variable_scope("feature_embedding") as scope:
            feature_embeddings = tf.contrib.layers.fully_connected(
                inputs=intermediate_layer,
                num_outputs=self.config.embedding_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                biases_initializer=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                scope=scope)
        self.feature_embeddings = feature_embeddings

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.
        Inputs:
          self.input_seqs
        Outputs:
          self.seq_embeddings
        """
        if self.mode == 'train':
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[self.config.batch_size, const_config.lstm_truncated_length],
                                        name="input_feed")
        else:
            # During inference, we always feed one word at a time.
            input_feed = tf.placeholder(dtype=tf.int64,
                                        shape=[self.config.batch_size],
                                        name="input_feed")

        with tf.variable_scope("seq_embedding"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=self.initializer)
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_feed)
        self.seq_embeddings = seq_embeddings

    def build_context_vector(self, prev_hidden_states, annotation_vectors):
        """
        Compute blending weights and build the context vector.
        :param prev_hidden_states: previous hidden states of shape [batch_size, num_lstm_units].
        :param annotation_vectors: frame level features of shape [batch_size, num_segments, input_feature_length]
        :return:
        """
        # The combining weights are computed via a multi-layer perceptron(MLP).
        with tf.variable_scope("build_context_vector", reuse=tf.AUTO_REUSE) as scope1:
            prev_hidden_states_tiled = tf.tile(tf.expand_dims(prev_hidden_states, axis=1), multiples=[1, self.config.num_segments, 1])
            prev_hidden_states_tiled = tf.reshape(prev_hidden_states_tiled, [-1, self.config.num_lstm_units])
            annotation_vectors_reshaped = tf.reshape(annotation_vectors, [-1, self.config.input_feature_size])
            num_hidden_units = self.config.input_feature_size + self.config.num_lstm_units
            input_feature = tf.concat([prev_hidden_states_tiled, annotation_vectors_reshaped], axis=1)
            intermediate_layer = tf.contrib.layers.fully_connected(
                inputs=input_feature,
                num_outputs=num_hidden_units,
                activation_fn=tf.nn.tanh,
                weights_initializer=self.initializer,
                biases_initializer=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                reuse=tf.AUTO_REUSE,
                scope=scope1)
            # Blending_weights is of shape [batch_size, num_segments].
            with tf.variable_scope("blending_weights", reuse=tf.AUTO_REUSE) as scope2:
                blending_weights = tf.contrib.layers.fully_connected(
                    inputs=intermediate_layer,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=self.initializer,
                    biases_initializer=None,
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                    biases_regularizer=tf.contrib.layers.l2_regularizer(self.config.regularization_strength),
                    reuse=tf.AUTO_REUSE,
                    scope=scope2)
            blending_weights = tf.reshape(blending_weights, [self.config.batch_size, self.config.num_segments])
            # Apply a softmax to convert the weights to [0,1].
            blending_weights = tf.nn.softmax(blending_weights)
            # annotation_vectors_reshaped is of shape [batch_size, num_segments, deep_feature_size].
            annotation_vectors_reshaped = tf.reshape(annotation_vectors,
                                                     [self.config.batch_size, self.config.num_segments, -1])
            # context_vector is of shape [batch_size, deep_feature_size].
            context_vector = tf.reduce_sum(tf.expand_dims(blending_weights, -1) * annotation_vectors_reshaped, axis=1)
        return context_vector

    def build_model(self):
        """Builds the model.

        Inputs:
          self.input_features
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)

        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """

        # Input feature
        target_sequences = tf.placeholder(dtype=tf.int32,
                                          shape=[self.config.batch_size, const_config.lstm_truncated_length],
                                          name="target_sequences")
        input_mask = tf.placeholder(dtype=tf.int32,
                                    shape=[self.config.batch_size, const_config.lstm_truncated_length],
                                    name="input_mask")

        with tf.variable_scope("lstm", initializer=self.initializer):
            # LSTM from scratch.
            global_video_feature_length = self.config.embedding_size
            word_embedding_size = self.config.embedding_size
            lstm_input_size = word_embedding_size + global_video_feature_length +\
                              self.config.num_lstm_units + self.config.input_feature_size
            W_forget = tf.get_variable(name="W_forget",
                                       shape=[lstm_input_size, self.config.num_lstm_units],
                                       initializer=self.initializer)
            b_forget = tf.get_variable(name="b_forget",
                                       initializer=tf.constant(
                                           np.ones(self.config.num_lstm_units, dtype=np.float32)))  # a nasty trick
            W_input = tf.get_variable(name="W_input",
                                      shape=[lstm_input_size, self.config.num_lstm_units],
                                      initializer=self.initializer)
            b_input = tf.get_variable(name="b_input",
                                      shape=[self.config.num_lstm_units],
                                      initializer=self.initializer)
            W_output = tf.get_variable(name="W_output",
                                       shape=[lstm_input_size, self.config.num_lstm_units],
                                       initializer=self.initializer)
            b_output = tf.get_variable(name="b_output",
                                       shape=[self.config.num_lstm_units],
                                       initializer=self.initializer)
            W_ctilde = tf.get_variable(name="W_tilde",
                                       shape=[lstm_input_size, self.config.num_lstm_units],
                                       initializer=self.initializer)
            b_ctilde = tf.get_variable(name="b_tilde",
                                       shape=[self.config.num_lstm_units],
                                       initializer=self.initializer)

            initial_state_tensor = \
                tf.get_variable(name="initial_state_tensor",
                                shape=[self.config.batch_size, self.config.num_lstm_units * 2],
                                initializer=self.initializer)

            if self.mode == "inference":
                tf.identity(initial_state_tensor, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.config.batch_size, self.config.num_lstm_units * 2],
                    name="state_feed")

                cell_state, hidden_state = tf.split(state_feed, num_or_size_splits=2, axis=1)

                # Run a single LSTM step.
                lstm_inputs = tf.concat([self.seq_embeddings, self.feature_embeddings], axis=1)
                context_vector = self.build_context_vector(hidden_state, self.input_features)
                input_state_concatenated = tf.concat([lstm_inputs, context_vector, hidden_state], axis=1)
                forget_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_forget) + b_forget)
                input_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_input) + b_input)
                output_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_output) + b_output)
                cell_state_tilde = tf.tanh(tf.matmul(input_state_concatenated, W_ctilde) + b_ctilde)
                cell_state = tf.multiply(forget_gate, cell_state) + tf.multiply(input_gate, cell_state_tilde)
                hidden_state = tf.multiply(output_gate, tf.tanh(cell_state))
                lstm_outputs = [hidden_state]

                # Concatenate the resulting state.
                tf.concat([cell_state, hidden_state], axis=1, name="state")
            else:
                tiled_feature_embeddings = tf.tile(
                    tf.expand_dims(self.feature_embeddings, 1),
                    (1, const_config.lstm_truncated_length, 1))
                lstm_inputs = tf.concat((self.seq_embeddings, tiled_feature_embeddings), axis=2)
                lstm_inputs = tf.unstack(lstm_inputs, axis=1)
                initial_cell_state, initial_hidden_state = \
                    tf.split(initial_state_tensor, num_or_size_splits=2, axis=1)
                cell_state = initial_cell_state
                hidden_state = initial_hidden_state
                lstm_outputs = []
                for i in range(const_config.lstm_truncated_length):
                    context_vector = self.build_context_vector(hidden_state, self.input_features)
                    input_state_concatenated = tf.concat([lstm_inputs[i], context_vector, hidden_state], axis=1)
                    forget_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_forget) + b_forget)
                    input_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_input) + b_input)
                    output_gate = tf.sigmoid(tf.matmul(input_state_concatenated, W_output) + b_output)
                    cell_state_tilde = tf.tanh(tf.matmul(input_state_concatenated, W_ctilde) + b_ctilde)
                    cell_state = tf.multiply(forget_gate, cell_state) + tf.multiply(input_gate, cell_state_tilde)
                    hidden_state = tf.multiply(output_gate, tf.tanh(cell_state))
                    lstm_outputs.append(hidden_state)

        # Stack batches vertically.
        # (lstm_outputs shape: [batch_size * lstm_truncated_length, state_size])
        lstm_outputs = tf.stack(lstm_outputs, axis=0)
        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.config.num_lstm_units])

        # Apply an FC layer for classification.
        # (logits shape: [batch_size * padded_size, vocabulary_size])
        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=self.config.vocab_size,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=logits_scope)

        # Apply softmax on scores to obtain class probabilities.
        tf.nn.softmax(logits, name="softmax")

        if self.mode == "train":
            # targets shape: (batch_size * padded_size)
            targets = tf.reshape(target_sequences, [-1])
            # weights shape: (batch_size * padded_size)
            weights = tf.to_float(tf.reshape(input_mask, [-1]))

            # Compute losses.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
            batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                                tf.reduce_sum(weights),
                                name="batch_loss")
            tf.losses.add_loss(batch_loss)
            regularization_losses = tf.losses.get_regularization_losses()
            regularization_loss = tf.reduce_sum(regularization_losses)

            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/regularization_loss", regularization_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_feature_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()
