"""Test the LSTM decoder model."""
import time
from lstm_decoder import *
from lstm_decoder_repeated_feed_image import *
import lstm_decoder_config as configuration
from data_loader import DataLoader
from vocabulary import Vocabulary
import const_config
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("validation_data_loader", "",
                       "Serialized data loader file for validation set.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Vocabulary file.")
tf.flags.DEFINE_string("model_path", "",
                       "Saved model path.")
tf.flags.DEFINE_string("output_file", "",
                       "Path the dump output json file.")
tf.flags.DEFINE_integer("repeated_feed_images", False,
                        "Repeated feed images to LSTM at each step.")
tf.flags.DEFINE_string("selection_method", "sampling",
                       "sampling/argmax/beam_search.")
tf.logging.set_verbosity(tf.logging.INFO)


def generate_dumb_batch(batch_size, num_segments):
    feature_size = 1024
    sigma = 0.1
    image_features = sigma * np.random.rand(batch_size, num_segments, feature_size)
    video_indices = None
    video_segment_indices = None
    valid_count = batch_size
    yield image_features, video_indices, video_segment_indices, valid_count


def remove_start_end_word_ids(word_id_array, vocab):
    result = []
    for word_id in word_id_array:
        if word_id == vocab.start_id:
            continue
        elif word_id == vocab.end_id:
            break
        else:
            result.append(word_id)
    return result


def main(args):
    assert FLAGS.validation_data_loader, "--vocab_file is required"
    assert FLAGS.vocab_file, "--vocab_file is required"
    assert FLAGS.model_path, "--model_path is required"
    assert FLAGS.selection_method in ['sampling', 'argmax', 'beam_search'],\
        "--selection_method can only be one of 'sampling', 'argmax' and 'beam_search'."
    model_config = configuration.ModelConfig()

    print('Loading vocabulary file...')
    vocab = Vocabulary(FLAGS.vocab_file)
    vocab_size = vocab.get_vocabulary_size()

    # Assign parameters to model configuration.
    model_config.vocab_size = vocab_size

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        print('Building LSTM decoder model for inference...')

        if not FLAGS.repeated_feed_images:
            model = LSTMDecoder(model_config, mode="inference")
        else:
            model = LSTMDecoderRepeatedImageFeed(model_config, mode="inference")
        model.build()

        print('Initializing variables...')
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        print('Loading saved model...')
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model_path)

        print('Initializing data loader for validation set...')
        start = time.time()
        data_loader_val = DataLoader()
        data_loader_val.load(FLAGS.validation_data_loader)
        end = time.time()
        time_elapsed = end - start
        print('Finished initializing data loader (time elapsed: %f)' % time_elapsed)

        print('Start inference...')
        initial_input_sequence = np.zeros(model_config.batch_size, dtype=np.int32)
        initial_input_sequence.fill(vocab.start_id)

        max_sentence_length = const_config.lstm_truncated_length + 1

        json_results = []
        for image_features, _, _, _, video_indices, video_segment_indices, valid_count in \
                data_loader_val.segmental_sampling_iter(batch_size=model_config.batch_size,
                                                        num_segments=model_config.num_segments):

            current_input = initial_input_sequence.copy()
            if not FLAGS.repeated_feed_images:
                current_state = sess.run(fetches="lstm/initial_state:0",
                                         feed_dict={"input_features:0": image_features})
            else:
                current_state = sess.run(fetches="lstm/initial_state:0",
                                         feed_dict={})

            generated_sentences =\
                np.zeros((model_config.batch_size, max_sentence_length), dtype=np.int32)
            generated_sentences[:, 0] = current_input
            completed_masks = np.zeros(model_config.batch_size, dtype=np.bool)

            for i in range(const_config.lstm_truncated_length):
                if not FLAGS.repeated_feed_images:
                    softmax_output, next_state = sess.run(
                        fetches=["softmax:0", "lstm/state:0"],
                        feed_dict={
                            "input_feed:0": current_input,
                            "lstm/state_feed:0": current_state
                        })
                else:
                    softmax_output, next_state = sess.run(
                        fetches=["softmax:0", "lstm/state:0"],
                        feed_dict={
                            "input_feed:0": current_input,
                            "lstm/state_feed:0": current_state,
                            "input_features:0": image_features
                        })

                if FLAGS.selection_method == 'sampling':
                    # Sample the next word according to the probability.
                    next_input = []
                    for probs in softmax_output:
                        next_input.append(np.random.choice(vocab_size, p=probs))
                    next_input = np.array(next_input)
                elif FLAGS.selection_method == 'argmax':
                    next_input = np.argmax(softmax_output, axis=1)
                else:
                    # TODO: implement beam search
                    next_input = None
                generated_sentences[:, i + 1] = next_input

                # Update input and state.
                current_input = next_input
                current_state = next_state

                # Early stop if we have generated the <END> token for all sentences.
                for j, word_id in enumerate(next_input):
                    if word_id == vocab.end_id:
                        completed_masks[j] = True
                if sum(completed_masks) == model_config.batch_size:
                    break

            # Extract text sentences.
            sentences = []
            for word_id_array in generated_sentences:
                word_id_array = remove_start_end_word_ids(word_id_array, vocab)
                text = vocab.id_array_to_sentence(word_id_array)
                sentences.append(text)
            sentences = sentences[:valid_count]

            for sentence in sentences:
                print sentence

            for i in range(valid_count):
                video_idx = video_indices[i]
                segment_idx = video_segment_indices[i]
                video = data_loader_val.videos[video_idx]
                video_segment = video.video_segments[segment_idx]
                caption_trimmed = remove_start_end_word_ids(video_segment.caption, vocab)
                gt_caption = vocab.id_array_to_sentence(caption_trimmed)
                video_segment_name = video.name + str(segment_idx)
                json_results.append({
                    'name': video_segment_name,
                    'video_caption': sentences[i],
                    'gt_caption': gt_caption
                })
        print('Finished Inference.')

        print('Dumping results...')
        fo = open(FLAGS.output_file, 'w')
        json.dump(json_results, fo, indent=4)
        fo.close()
        print('Done.')


if __name__ == "__main__":
    tf.app.run()
