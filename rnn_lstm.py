from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
import os.path
import time
import collections
from sklearn.externals import joblib
import corpuslib as cpl
import sys
import math

from tensorflow.models.rnn.ptb import reader

import matplotlib.pyplot as plt

num_steps = 19
learning_rate = 1
embedding_size = 300
vocabulary_size = 30000
state_size = 1024
batch_size = 25
train_steps = 100000
eval_steps = 10
eval_batch_size = 2000
learning_rate = 2e-5
print_freq = 100
eval_freq = 200

dt = tf.float32

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
BATCH_SIZE_DEFAULT = 100
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
### --- END default constants---

FLAGS = None

class BatchCreator(object):
    """Batch creator class. Shuffles once."""

    def __init__(self, seqs):
        """Initialise."""
        self.seqs = np.copy(seqs)
        self.size = seqs.shape[0]
        np.random.shuffle(self.seqs)
        self.index = 0

    def create(self, batch_size):
        if self.index + batch_size >= self.size-1:
            np.random.shuffle(self.seqs)
            self.index = 0
        self.index += batch_size
        seqs = self.seqs[self.index:self.index+batch_size]
        return seqs[:, :-1], seqs[:, 1:]

def build_tf_graph(init_embedding):
    '''
    Computes the TF graph for a RNN language model using truncated backpropagation through time.

            init_embedding:        the initial embedding
            vocab_size:                size of the input vocabulary (number of word types)
            batch_size:                size of a mini batch
            num_steps:                 number of steps to backpropagate the error
            state_size:                size of the hidden state of the RNN cell
    '''
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units=state_size, state_is_tuple=True)

    # placeholders for input and targets
    x = tf.placeholder(tf.int32, [None, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, num_steps], name='output_placeholder')
    dropout_placeholder = tf.placeholder(tf.bool, name='dropout_placeholder')
    batch_size = tf.shape(x)[0]

    # init state for the rnn cell
    init_state = (tf.zeros([batch_size, state_size], tf.float32),
                  tf.zeros([batch_size, state_size], tf.float32))

    # embed the vector
    initial_embedding = tf.constant(init_embedding, name="initial_embedding")
    embedding = tf.get_variable("embedding", initializer=initial_embedding, dtype=dt)
    embedded_vectors = tf.nn.embedding_lookup(embedding, x)

    # squeeze all inputs into a sequence for the rnn cell
    #x_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, btt_steps, x_one_hot)]
    x_as_list = [tf.squeeze(i, squeeze_dims=[1])
                             for i in tf.split(1, num_steps, embedded_vectors)]

    # appply the rnn cell to the sequence of input vectors
    rnn_in = x_as_list
    rnn_out, training_state = tf.nn.rnn(rnn_cell, rnn_in, initial_state=init_state)

    # project the output sequence via a softmax layer back to vocabulary space
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [vocabulary_size, rnn_cell.output_size])
        b = tf.get_variable('b', [vocabulary_size], initializer=tf.constant_initializer(0.0))


    rnn_out_flat = tf.reshape(rnn_out, [-1, rnn_cell.output_size])
    labels_flat = tf.reshape(y, [-1])
    logits = tf.matmul(rnn_out_flat, tf.transpose(W)) + b
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels_flat)

    seq_losses = tf.reduce_mean(tf.reshape(losses, [-1, num_steps]), 1)
    perplexities = tf.exp(seq_losses)
    avg_perplexity = tf.reduce_mean(perplexities)

    total_loss = tf.reduce_mean(losses)

    # optimize all free variables with respect to that using Adam optimiziation (faster than SGD or Adagrad)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # return nodes of the tf graph which are used for further procesing
    return {
        'total_loss': total_loss,
        'x': x,
        'y': y,
        'init_state': init_state,
        'training_state': training_state,
        'train_step': train_step,
        'zero_state': init_state,
        'avg_perplexity': avg_perplexity
    }


#This function trains the network on the given dataset for a number of epochs.
#
#__TODO__: currently the learned parameters or not returned ye
def train_network(init_embedding, data, test_data, log_dir):
    '''
            Trains the RNN language model and prints the current average loss per sequence.

                    num_epochs:         number of iterations over the full corpus
                    data:                     input data (list of token indices)
                    vocab_size:         size of the input vocabulary (number of word types)
                    batch_size:         size of a mini batch
                    num_steps:            length of a token sequence
                    btt_steps:            number of steps to backpropagate the error
                    state_size:         size of the hidden state of the RNN cell
    '''

    with tf.Graph().as_default(), tf.Session() as sess:
        g = build_tf_graph(init_embedding)

        init = tf.initialize_all_variables()
        sess.run(init)

        losses = []
        batch_creator = BatchCreator(data)
        writer = tf.train.SummaryWriter(log_dir, graph=sess.graph)
        average_loss = 0
        average_time = 0
        for step in range(train_steps):
            _x, _y = batch_creator.create(batch_size)
            start_time = time.time()

            _loss, _ = sess.run(
                (g['total_loss'], g['train_step']),
                feed_dict={
                    g['x']: _x,
                    g['y']: _y})

            average_loss += _loss
            average_time += time.time() - start_time

            if step % print_freq == 0 or step == train_steps-1:
                if step > 0:
                    average_loss /= print_freq
                    average_time /= print_freq
                template = "Average loss at step {}: {:.3f} ({:.1f}ms)"
                print(template.format(
                    step, average_loss, average_time*1000))
                average_loss = 0
                average_time = 0

            if step % eval_freq == 0 or step == train_steps-1:
                eval_batch_creator = BatchCreator(test_data)
                start_time = time.time()
                cum_loss = 0.
                cum_perplexity = 0.
                for _ in range(eval_steps):
                    eval_inputs, eval_labels = eval_batch_creator.create(
                        eval_batch_size)
                    eval_feed_dict = {
                        g['x']: eval_inputs,
                        g['y']: eval_labels}
                    batch_loss, batch_perplexity = sess.run(
                        [g['total_loss'], g['avg_perplexity']],
                        feed_dict=eval_feed_dict)
                    cum_loss += batch_loss
                    cum_perplexity += batch_perplexity
                eval_loss = cum_loss / eval_steps
                eval_perplexity = cum_perplexity / eval_steps
                duration = time.time() - start_time
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="test-loss", simple_value=eval_loss),
                ])
                writer.add_summary(summary, step)
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="test-perplexity", simple_value=eval_perplexity),
                ])
                writer.add_summary(summary, step)
                template = "Evaluated test at step {}: loss={:.3f} perplexity={:.3f} ({:.1f}s)"
                print(template.format(
                    step, eval_loss, eval_perplexity, duration))

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    """
    Main function
    """

    #FLAGS.log_dir = FLAGS.log_dir + '/dummy'

    # Print all Flags to confirm parameter settings
    # print_flags()

    data = cpl.load_data(data_path='parsed/indices.npy')
    zipf_idxs, inv_zipf_idx, counts = cpl.create_zipf_index(data)
    data = cpl.zipf(data, zipf_idxs)
    data = cpl.unknownize(data, vocabulary_size)
    seqs = cpl.to_sequences(data)
    packet = joblib.load('sample_pickle/sample_pickle.pkl')
    oov_widxs, (oov_seqs_idxs, iv_seqs_idxs), \
        (oov_seqs_train_idxs, oov_seqs_test_idxs,
         iv_seqs_train_idxs, iv_seqs_test_idxs) = packet

    # Run the training operation
    job = sys.argv[1]
    log_dir = 'logs/rnn/'+job
    if job == 'nopretrain':
        init_embedding = np.random.normal(
            size=[vocabulary_size, embedding_size], scale=1.0/math.sqrt(embedding_size)).astype(np.float32)
    elif job == 'baseline':
        init_embedding, _ = joblib.load("init_embeddings/baseline")
        init_embedding = init_embedding.astype(np.float32)
    else:
        init_embedding, _ = joblib.load("init_embeddings/trained-"+job)
        init_embedding = init_embedding.astype(np.float32)
    train_network(init_embedding, seqs[oov_seqs_train_idxs], seqs[oov_seqs_test_idxs], log_dir)

if __name__ == '__main__':
    tf.app.run()
