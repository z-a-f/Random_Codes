#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections

import tensorflow as tf

from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
# from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib import layers

from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
from tensorflow.python.ops import array_ops

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
