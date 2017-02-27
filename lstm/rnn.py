import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear

from . import lstm

def RNN(x, weights, biases, n_input = 1, n_steps = 1, n_hidden = 16):
  # Prepare data shape to match `rnn` function requirements
  # Current data input shape: (batch_size, n_steps, n_input)
  # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

  # Permuting batch_size and n_steps
  x = tf.transpose(x, [1, 0, 2])
  # Reshaping to (n_steps*batch_size, n_input)
  x = tf.reshape(x, [-1, n_input])
  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  x = tf.split(x, n_steps, 0)

  # Define a lstm cell with tensorflow
  lstm_cell = lstm.LSTMCellOlah(n_hidden)

  # Get lstm cell output
  outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']
