"""
The code implements LSTM decoder module used for caption generation.

The basic idea is the same with show-and-tell model.

The code is modified from show-and-tell source code https://github.com/tensorflow/models/tree/master/research/im2txt.
"""

import tensorflow as tf
import const_config


class LSTMDecoder(object):
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
        input_features = tf.placeholder(dtype=tf.float32,
                                        shape=[self.config.batch_size,
                                               self.config.num_segments,
                                               self.config.input_feature_size],
                                        name="input_features")
        reshaped_feature = tf.reshape(input_features, (self.config.batch_size, -1))
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

        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, forget_bias=0.0)
             for _ in range(self.config.num_lstm_layers)], state_is_tuple=True)

        if self.mode == "train":
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob)

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            # Create a all zero state for the LSTM cell (zero_state shape: [num_layer, 2, batch_size, state_size]).
            zero_state = lstm_cell.zero_state(batch_size=self.feature_embeddings.get_shape()[0], dtype=tf.float32)

            # Feed the input feature to set the initial LSTM state.
            # (initial_state shape: [num_layer, 2, batch_size, state_size]).
            _, initial_state = lstm_cell(self.feature_embeddings, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            if self.mode == "inference":
                # In inference mode, use concatenated states for convenient feeding and fetching.
                # initial_state is of shape (num_layer, 2, batch_size, state_size)
                tf.stack(axis=0, values=initial_state, name="initial_state")

                # Placeholder for feeding a batch of concatenated states.
                state_feed = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.config.num_lstm_layers, 2, self.config.batch_size, self.config.num_lstm_units],
                    name="state_feed")

                per_layer_state_tuple = [tf.contrib.rnn.LSTMStateTuple(state_feed[idx][0], state_feed[idx][1])
                                         for idx in range(self.config.num_lstm_layers)]

                # Run a single LSTM step.
                # During inference, the dimension at index 1 must be 1.
                # Because each during each inference step we only feed one word.
                # lstm_outputs shape: (batch_size, state_size)
                # state_tuple: [(batch_size, state_size[0]), (batch_size, state_size[1])]
                lstm_outputs, state_tuple = lstm_cell(
                    inputs=self.seq_embeddings,
                    state=per_layer_state_tuple)

                # Concatentate the resulting state.
                tf.stack(values=state_tuple, name="state")
            else:
                # Compute the length of each sequence in the mini-batch.
                sequence_length = tf.reduce_sum(input_mask, 1)

                # Run the batch of sequence embeddings through the LSTM.
                # (lstm_outputs shape: [batch_size, padded_size, state_size])
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                    inputs=self.seq_embeddings,
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)

        # Stack batches vertically.
        # (lstm_outputs shape: [batch_size * padded_size, state_size])
        lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

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

