import numpy as np
import tensorflow as tf
import nltk.tokenize


class Vocabulary(object):
    """Vocabulary class for the LSTM decoder model."""

    def __init__(self,
                 vocab_file,
                 start_word="<START>",
                 end_word="</END>",
                 unk_word="<UNKNOWN>"):
        """Initializes the vocabulary.
        Args:
          vocab_file: File containing the vocabulary, where the words are the first
            whitespace-separated token on each line (other tokens are ignored) and
            the word ids are the corresponding line numbers.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        assert start_word in reverse_vocab
        assert end_word in reverse_vocab
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_word = start_word
        self.end_word = end_word
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]

    def sentence_to_id_array(self, sentence):
        """Returns the integer word id array of a sentence."""
        word_ids = []
        words = nltk.tokenize.word_tokenize(sentence.lower())
        if words[0] != self.start_word:
            words.insert(0, self.start_word)
        if words[-1] != self.end_word:
            words.append(self.end_word)
        for word in words:
            word_ids.append(self.word_to_id(word))
        return np.array(word_ids)

    def id_array_to_sentence(self, word_ids):
        """Returns the sentence of an integer word id array."""
        words = []
        for word_id in word_ids:
            words.append(self.id_to_word(word_id))
            if word_id == self.end_id:
                break
        sentence = ' '.join(words)
        return sentence

    def get_vocabulary_size(self):
        # Plus one because we didn't include <UNKNOWN>
        return len(self.vocab) + 1
