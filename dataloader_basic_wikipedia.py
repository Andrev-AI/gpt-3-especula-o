import tensorflow as tf
import numpy as np

class WikipediaDataLoader():
    def __init__(self, data_path, batch_size, seq_length, vocab_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.char_to_index = {char: index for index, char in enumerate(self.chars)}

        self.text_as_int = np.array([self.char_to_index[char] for char in self.text])

        self.num_batches = len(self.text_as_int) // (self.batch_size * self.seq_length)

        self.text_as_int = self.text_as_int[:self.num_batches * self.batch_size * self.seq_length]

        self.text_as_int = self.text_as_int.reshape([self.batch_size, -1, self.seq_length])

    def get_batch(self, idx):
        # Get batch of data at specified index
        x = self.text_as_int[:, idx * self.seq_length: (idx + 1) * self.seq_length]
        y = np.zeros_like(x)


        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

        # Convert data to one-hot encoding
        x_one_hot = tf.one_hot(x, depth=self.vocab_size)
        y_one_hot = tf.one_hot(y, depth=self.vocab_size)

        return x_one_hot, y_one_hot
