"""LSTM decoder model and training configurations."""


class ModelConfig(object):
    """Wrapper class for model hyper-parameters."""

    def __init__(self):
        """Sets the default model hyper-parameters."""

        # Number of unique words in the vocab (including <START>, <END>, <UNKNOWN>).
        self.vocab_size = 0

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # Batch Size
        self.batch_size = 512

        # Number of segments used in sampling training features for each video.
        # Number of frames used to represent each video
        self.num_segments = 5

        # Input feature dimensionality
        self.input_feature_size = 1024

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = 256
        self.num_lstm_units = 256

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.5


class TrainingConfig(object):
    """Wrapper class for training hyper-parameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        # Total Number of Frames (1381336) / Number of Segments (5)
        self.num_examples_per_epoch = 276267

        # Optimizer for training the model.
        self.optimizer = "SGD"

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.001
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 1.0

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 3

        # Number of training iterations.
        self.num_iterations = -1

        # Batch size used for SGD.
        self.batch_size = 32

        # Log every N steps.
        self.log_every_n_steps = 1

        # Save model every N steps
        self.save_every_n_steps = 5000

        # Directory for saving and loading model checkpoints.
        self.train_dir = ""

        # Compute evaluation loss every N steps
        self.validation_loss_every_n_steps = 5
