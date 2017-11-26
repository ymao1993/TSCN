"""Train the LSTM decoder model."""
import time
from lstm_decoder import *
import lstm_decoder_config as configuration
from data_loader import DataLoader
from vocabulary import Vocabulary
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("training_data_loader", "",
                       "Serialized data loader file for training set.")
tf.flags.DEFINE_string("validation_data_loader", "",
                       "Serialized data loader file for validation set.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Vocabulary file.")
tf.flags.DEFINE_string("train_dir", "train_dir",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("summary_dir", "summary_dir",
                       "Directory for saving summaries.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("validation_loss_every_n_steps", 10,
                        "Frequency at which validation loss is computed for one mini-batch.")
tf.logging.set_verbosity(tf.logging.INFO)


def generate_dumb_batch(batch_size, num_segments):
    feature_size = 1024
    sigma = 0.1
    image_features = sigma * np.random.rand(batch_size, num_segments, feature_size)
    input_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))
    input_mask = np.ones(shape=(batch_size, const_config.lstm_truncated_length), dtype=np.int)
    target_sequence = np.zeros(shape=(batch_size, const_config.lstm_truncated_length))
    return image_features, input_sequence, input_mask, target_sequence


def main(args):
    assert FLAGS.training_data_loader, "--training_data_loader is required"
    assert FLAGS.vocab_file, "--vocab_file is required"
    assert FLAGS.train_dir, "--train_dir is required"

    model_config = configuration.ModelConfig()
    training_config = configuration.TrainingConfig()

    print('Loading vocabulary file...')
    vocab = Vocabulary(FLAGS.vocab_file)
    vocab_size = vocab.get_vocabulary_size()

    # Assign parameters to model configuration.
    model_config.vocab_size = vocab_size
    training_config.batch_size = FLAGS.batch_size
    training_config.train_dir = FLAGS.train_dir
    training_config.num_iterations = FLAGS.number_of_steps
    training_config.log_every_n_steps = FLAGS.log_every_n_steps
    training_config.validation_loss_every_n_steps = FLAGS.validation_loss_every_n_steps

    # Create training directory.
    if not tf.gfile.IsDirectory(training_config.train_dir):
        tf.logging.info("Creating training directory: %s", training_config.train_dir)
        tf.gfile.MakeDirs(training_config.train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        print('Building LSTM decoder model...')
        model = LSTMDecoder(model_config, mode="train")
        model.build()

        print('Initializing data loader for training set...')
        start = time.time()
        data_loader_train = DataLoader()
        data_loader_train.load(FLAGS.training_data_loader)
        end = time.time()
        time_elapsed = end - start
        print('Finished initializing data loader (time elapsed: %f)' % time_elapsed)

        print('Initializing data loader for validation set...')
        start = time.time()
        data_loader_val = DataLoader()
        data_loader_val.load(FLAGS.validation_data_loader)
        end = time.time()
        time_elapsed = end - start
        print('Finished initializing data loader (time elapsed: %f)' % time_elapsed)

        # Setup learning rate decay.
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(
            training_config.initial_learning_rate, global_step,
            decay_steps=decay_steps,
            decay_rate=training_config.learning_rate_decay_factor,
            staircase=True)

        # Setup optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(model.total_loss, global_step=global_step)

        # Setup summary.
        all_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
        val_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/val')

        # Create saver
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

        # Initialize variables.
        print('Initializing variables...')
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        print('Start training...')
        # Stochastic Gradient Descent
        for i in range(training_config.num_iterations):
            print('Sampling mini-batch...')
            image_features, input_sequence, input_mask, target_sequence =\
                data_loader_train.segmental_sampling(batch_size=training_config.batch_size,
                                                     num_segments=model_config.num_segments)

            _, total_loss, summary = sess.run((train, model.total_loss, all_summary),
                                              feed_dict={"input_features:0": image_features,
                                                         "input_feed:0": input_sequence,
                                                         "input_mask:0": input_mask,
                                                         "target_sequences:0": target_sequence})
            train_writer.add_summary(summary, i)

            # Logging
            if i % training_config.log_every_n_steps == 0:
                print('[%d/%d] loss: %f' % (i, training_config.num_iterations, total_loss))

            # Save model.
            if i % training_config.save_every_n_steps == 0:
                print('Saving model at step %d...' % i)
                saver.save(sess, FLAGS.train_dir + '/model', global_step=i)

            # evaluate the loss with validation set at every epoch
            if i % training_config.validation_loss_every_n_steps == 0:
                image_features, input_sequence, input_mask, target_sequence = \
                    data_loader_val.segmental_sampling(batch_size=training_config.batch_size,
                                                       num_segments=model_config.num_segments)

                total_loss, summary = sess.run((model.total_loss, all_summary),
                                               feed_dict={"input_features:0": image_features,
                                                          "input_feed:0": input_sequence,
                                                          "input_mask:0": input_mask,
                                                          "target_sequences:0": target_sequence})
                val_writer.add_summary(summary, i)


if __name__ == "__main__":
    tf.app.run()
