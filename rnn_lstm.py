from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import numpy as np
import os.path
import time
import collections

from tensorflow.models.rnn.ptb import reader

import matplotlib.pyplot as plt

num_steps = 21
num_epochs = 10
embedding_size = 99
state_size = 77 #multiply times 4 for the use of LSTM
num_negative_samples = 40

dt = tf.float32

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
BATCH_SIZE_DEFAULT = 83
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
### --- END default constants---

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './ptb_data')
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/'

FLAGS = None

def get_data():
    '''
        Returns the Penn Tree Bank corpus as list of indices, each representing a unique words type in the corpus.
    '''
    raw_data = reader.ptb_raw_data(FLAGS.data_dir)
    return tuple(map(np.array, raw_data[:3])), raw_data[-1]

def reindex_sorted(data):
    unique, counts = np.unique(data, return_counts=True)
    index = unique[np.argsort(counts)[::-1]]
    m = { index[i]: i for i in range(len(unique))  }
    m_inv = { i: index[i] for i in range(len(unique))  }
    transform = np.vectorize(lambda x: m[x], otypes=[np.int32])
    return transform(data), m, m_inv

(train_data, valid_data, test_data), vocab_size = get_data()
reindex_sorted(np.array([2,4,5,5,5,5,56,6,6,6,6,6,6,6,6,6,6,6,6,6,6,4,3,2,3,4,2,2,2,3,4,2]))

n_truncate = 100000000
train_data = train_data[:n_truncate]
valid_data = train_data[:n_truncate]
test_data = train_data[:n_truncate]

#These two functions generate the epochs of batches for the training. Both implement the iterator pattern.
def generate_batch(data, vocab_size, batch_size, num_steps):
    '''
        Generates a mini batch of token sequences. 
        
            data:           input data (list of token indices)
            vocab_size:     size of the input vocabulary (number of word types)
            batch_size:     size of a mini batch
            num_steps:      length of a token sequence
    '''
    begins = np.random.randint(vocab_size - num_steps - 1, size=batch_size)[:,np.newaxis]
    ranges = np.arange(num_steps, dtype=np.int)[np.newaxis,:]
    indices = begins + ranges    
    return data[indices], data[indices+1]


def generate_epoch(data, vocab_size, batch_size, num_steps):
    num_batches = int(len(data) / batch_size / num_steps)
    for j in range(num_batches):
        yield generate_batch(data, vocab_size, batch_size, num_steps), num_batches

def build_tf_graph(init_embedding, vocab_size, batch_size, num_steps, state_size, num_negative_samples=6):
    '''
        Computes the TF graph for a RNN language model using truncated backpropagation through time.
        
            init_embedding:    the initial embedding
            vocab_size:        size of the input vocabulary (number of word types)
            batch_size:        size of a mini batch
            num_steps:         number of steps to backpropagate the error
            state_size:        size of the hidden state of the RNN cell
    '''
    
    # placeholders for input and targets
    x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, num_steps], name='output_placeholder')
    dropout_placeholder = tf.placeholder(tf.bool, name='dropout_placeholder')

    # init state for the rnn cell
    default_init_state = tf.zeros([batch_size, state_size])
    init_state = tf.placeholder_with_default(default_init_state, [batch_size, state_size], 
                                             name='state_placeholder')
    
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
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
    rnn_out, training_state = tf.nn.rnn(rnn_cell, rnn_in, initial_state=init_state)
    
    # project the output sequence via a softmax layer back to vocabulary space
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [vocab_size, state_size])
        b = tf.get_variable('b', [vocab_size], initializer=tf.constant_initializer(0.0))
    
    losses = tf.nn.nce_loss(
        weights=W,
        biases=b,
        inputs=tf.reshape(rnn_out, [batch_size * num_steps, state_size]),
        labels=tf.reshape(y, [batch_size * num_steps, 1]),
        num_sampled=num_negative_samples,
        num_classes=vocab_size,
        remove_accidental_hits = True
    )

    total_loss = tf.reduce_mean(losses)
    
    # optimize all free variables with respect to that using Adam optimiziation (faster than SGD or Adagrad)
    train_step = tf.train.AdamOptimizer().minimize(total_loss)
    
    # return nodes of the tf graph which are used for further procesing
    return {
        'total_loss': total_loss,
        'x': x,
        'y': y,
        'init_state': init_state,
        'training_state': training_state,
        'train_step': train_step
    }  


#This function trains the network on the given dataset for a number of epochs.
#
#__TODO__: currently the learned parameters or not returned ye
def train_network(init_embedding, num_epochs, data, vocab_size, 
                  batch_size, num_steps, state_size, num_negative_samples):
    '''
        Trains the RNN language model and prints the current average loss per sequence.
        
            num_epochs:     number of iterations over the full corpus
            data:           input data (list of token indices)
            vocab_size:     size of the input vocabulary (number of word types)
            batch_size:     size of a mini batch
            num_steps:      length of a token sequence
            btt_steps:      number of steps to backpropagate the error
            state_size:     size of the hidden state of the RNN cell 
    '''
    
    with tf.Graph().as_default(), tf.Session() as sess:
        g = build_tf_graph(init_embedding.astype(np.float32), vocab_size, 
                           batch_size, num_steps, state_size, num_negative_samples)

        init = tf.global_variables_initializer()
        sess.run(init)

        losses = []
        for k in range(num_epochs):
            epoch = generate_epoch(data, vocab_size, batch_size, num_steps)
            _agg_loss = 0
            for i, ((_x, _y), num_batches) in enumerate(epoch):
                
                _state = np.zeros((batch_size, state_size))
                _loss, _state, _ = sess.run(
                    (
                        g['total_loss'], 
                        g['training_state'],
                        g['train_step']
                    ),
                    feed_dict={
                        g['x']: _x,
                        g['y']: _y,
                        g['init_state']: _state
                    }
                )
                
                _agg_loss += _loss
                losses.append(_agg_loss / (i+1))
                
                print("\repoch:",k+1,"/",num_epochs, end="")
                print(" batch:", i+1,"/", num_batches,
                      "avg loss", _agg_loss / (i+1), end="")

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
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  init_embedding = np.random.uniform(size=(vocab_size, embedding_size)).astype(np.float32)
  train_network(init_embedding, num_epochs, train_data, vocab_size, FLAGS.batch_size, 
              num_steps, state_size, num_negative_samples)

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

tf.app.run()
