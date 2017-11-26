""" Build vocabulary.
"""

from collections import Counter
import nltk.tokenize
import tensorflow as tf
import sys
sys.path.insert(0, '../')
from data_manager import *

tf.flags.DEFINE_string("caption_json", "",
                       "ActivityNet Caption JSON file.")
tf.flags.DEFINE_string("start_word", "<START>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</END>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNKNOWN>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "data/word_counts.txt",
                       "Output vocabulary file of word counts.")
tf.flags.DEFINE_string("vocabulary_file", "data/vocab.txt",
                       "Output vocabulary file of word-id mappings.")
FLAGS = tf.flags.FLAGS


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id


def create_vocab(captions):
    """Creates the vocabulary of word to word_id.

    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.

    Args:
      captions: A list of lists of strings.

    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words (including begin and end tokens, but excluding unknown words):", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary (including begin and end tokens, but excluding unknown words):", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote word count file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
    vocab_dict = dict([(word, idx) for (idx, word) in enumerate(reverse_vocab)])

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.vocabulary_file, "w") as f:
        f.write("\n".join(["%s %d" % (word, idx) for word, idx in vocab_dict.iteritems()]))
    print("Wrote vocabulary file:", FLAGS.vocabulary_file)


def process_caption(caption):
    """Processes a caption string into a list of tonenized words.

    Args:
      caption: A string caption.

    Returns:
      A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def main(args):
    print('Initializing data manager from %s...' % FLAGS.caption_json)
    data_manager = DataManager()
    data_manager.load_captions(FLAGS.caption_json)

    print('Extracting captions...')
    captions = data_manager.raw_captions
    captions = list(set(captions))

    print('Tokenizing captions...')
    tokenized_captions = []
    for caption in captions:
        tokenized_captions.append(process_caption(caption))

    print('Creating word count file and vocabulary file...')
    create_vocab(tokenized_captions)

if __name__ == "__main__":
    tf.app.run()
