#!/usr/bin/env python

from __future__ import print_function

import collections

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.python.ops import array_ops

# from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib import layers
# from tensorflow.python.ops.core_rnn_cell_impl import _linear as linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import collections
_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class LSTMCellOlah(tf.contrib.rnn.RNNCell):
  """
  See the following:
    - http://arxiv.org/abs/1409.2329
    - http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    - http://karpathy.github.io/2015/05/21/rnn-effectiveness/

  This is a simplified implementation -- it ignores most biases, and it somewhat simplifies the equations shown in the references above.
  """
  def __init__(self, num_units, *args, **kwargs):
    self._num_units = num_units

  @property
  def state_size(self):
    return LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope="LSTM"):
    with tf.variable_scope(scope):
      c, h = state

      concat = linear([inputs, h], 4 * self._num_units, True)
      # gates = layers.fully_connected(tf.concat(1, [inputs, h]),
      #                                 num_outputs=4 * self._num_units,
      #                                 activation_fn=None)
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      forget_bias = 1.0
      new_c = (c * tf.nn.sigmoid(f + forget_bias)
               + tf.nn.sigmoid(i) * tf.nn.tanh(j))
      new_h =  tf.nn.tanh(new_c) * tf.nn.sigmoid(o)

      return new_h, LSTMStateTuple(new_c, new_h)

  def zero_state(self, batch_size, dtype=tf.float32, learnable=False, scope="LSTM"):
    if learnable:
      c = tf.get_variable("c_init", (1, self._num_units),
              initializer=tf.random_normal_initializer(dtype=dtype))
      h = tf.get_variable("h_init", (1, self._num_units),
              initializer=tf.random_normal_initializer(dtype=dtype))
    else:
      c = tf.zeros((1, self._num_units), dtype=dtype)
      h = tf.zeros((1, self._num_units), dtype=dtype)
    c = tf.tile(c, [batch_size, 1])
    h = tf.tile(h, [batch_size, 1])
    c.set_shape([None, self._num_units])
    h.set_shape([None, self._num_units])
    return (c, h)


'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

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
    # lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = LSTMCellOlah(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
