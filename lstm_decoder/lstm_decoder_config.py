"""LSTM decoder model and training configurations."""

default_batch_size = 256


class ModelConfig(object):
    """Wrapper class for model hyper-parameters."""

    def __init__(self):
        """Sets the default model hyper-parameters."""

        # Number of unique words in the vocab (including <START>, <END>, <UNKNOWN>).
        self.vocab_size = 0

        # Batch Size
        self.batch_size = default_batch_size

        # Number of segments used in sampling training features for each video.
        # Number of frames used to represent each video
        self.num_segments = 8

        # Input feature dimensionality
        self.input_feature_size = 1024

        # Number of LSTM layers
        self.num_lstm_layers = 2

        # Number of units in the intermediate FC layer for converting input feature to feature embedding.
        self.num_units_intermediate_fc = 1024

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 512
        self.num_lstm_units = 1024

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.95

        # Regularization strength applied to image image embedding layer
        self.regularization_strength = 0.000001


class TrainingConfig(object):
    """Wrapper class for training hyper-parameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        # Total Number of Frames (1381336) / Number of Segments (5)
        self.num_examples_per_epoch = 138134

        # Optimizer for training the model.
        self.optimizer = "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 10.0  # prev: 2.0

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 3

        # Number of training iterations.
        self.num_iterations = -1

        # Batch size used for SGD.
        self.batch_size = default_batch_size

        # Log every N steps.
        self.log_every_n_steps = 1

        # Save model every N steps
        self.save_every_n_steps = 1000

        # Directory for saving and loading model checkpoints.
        self.train_dir = ""

        # Compute evaluation loss every N steps
        self.validation_loss_every_n_steps = 5
