"""
Module provides ability to create word embeddings and examine training process.

Much of the code is employed from Tensorflow's word embedding tutorial.
https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
"""

import math
import random
import collections

import numpy as np
import tensorflow as tf


class WordEmbedder:
    """Handles a corpus of words and provides their embedding"""

    def __init__(self, words, n_words):
        self.vocabulary_size = n_words
        self.data, self.count, self.dictionary, self.reverse_dictionary = self.provide_dataset(words, n_words)
        self.data_index = 0
        self.batch, self.labels = self.generate_batch(batch_size=8, num_skips=2, skip_window=1)

    def provide_dataset(self, words, n_words):
        """Process raw inputs. Returns a dataset"""
        count = [['UNK', -1]]
        # Count the most common words in the list provided.
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
            # As the most common words in count are entered first, they will receive
            # a higher value (i.e. closer to 1) in the dictionary, allowing us to
            # use this to select for these common words later
        data = list()
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)  # return 0 if word not in dictionary
            if index == 0:  # i.e. dictionary['UNK'] or infrequent words
                unk_count += 1
            data.append(index)  # Create a list of all the indices
        count[0][1] = unk_count  # Change UNK count from 0 to the number of unknown (UNK) words in the dataset

        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    def generate_batch(self, batch_size, num_skips, skip_window):
        # using assert for debugging. If batch_size not perfectly divisible
        # by num_skips then an error will be raised.
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # Create a span of words which covers your target word and the
        # context words around it
        span = 2 * skip_window + 1
        # deque is like a list but allows appending and popping from either side
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        # Add to buffer data from some index to the span length, defined by skip_window
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span  # moves data_index along by 'span' units
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]  # selects word of interest
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])  # using append as we're just adding a single value.
                self.data_index += 1
        # Backtrack to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def train(self, target):
        target_words = [self.dictionary.get(word) for word in target]
        target_size = len(target)
        batch_size = 128
        embedding_size = 128  # Dimension of the embeding vector
        skip_window = 1  # How many words to consider left and right
        num_skips = 2  # How many times to reuse an input to generate a label
        num_sampled = 64  # Number of negative examples to sample
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        target_dataset = tf.constant(target_words, dtype=tf.int32)

        # Create random embeddings to begin with for each word in vocabulary, with a dimension of embedding_size
        embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # embedding_lookup simply gets the embeding vector for each row (word in vocab) which is given by train_inputs

        # Construct the variable for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal(
                [self.vocabulary_size, embedding_size],
                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=num_sampled,
                num_classes=self.vocabulary_size))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        target_embeddings = tf.nn.embedding_lookup(normalized_embeddings, target_dataset)

        # All this most frequent words "the" "and" "it" should appear in the same
        # contexts and be very similar. Therefore we can use these to see how well
        # our model is doing
        similarity = tf.matmul(target_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer
        init = tf.global_variables_initializer()

        # BEGIN TRAINING
        num_steps = 100001

        # Add the loss value as a scalar to summary
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with tf.Session() as session:
            # We must initialise all variables before we use them
            init.run()
            print('Initialized')

            average_loss = 0
            for step in range(num_steps):
                # Use our function from before to generate small batches of words
                # to train our model on
                batch_inputs, batch_labels = self.generate_batch(batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op
                # (including it in the list of retrned values for session.run())
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable
                _, loss_val = session.run(
                    [optimizer, loss],
                    feed_dict=feed_dict)
                average_loss += loss_val  # keep tally of loss calculated

                # if step % 2000 == 0:
                #     if step > 0:
                #         average_loss /= 2000
                #     # Use tally of loss calculated to work out the average loss
                #     # for every 2000 steps
                #     print(f'Average losss at step {step}: {average_loss}')
                #     average_loss = 0

                # Every 10000 steps show progress of the training by demonstrating
                # which words are close to the validation words set previously
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(target_size):
                        target_word = self.reverse_dictionary[target_words[i]]
                        top_k = 8  # number of nearest neigbours
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = f'Nearest to {target_word}:'
                        for k in range(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log_str = f'{log_str} {close_word}'
                        print(log_str)
            return normalized_embeddings.eval()
