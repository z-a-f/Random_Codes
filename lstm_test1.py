#!/usr/bin/env python

from __future__ import print_function

import collections
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf

# from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib import layers
# from tensorflow.python.ops.core_rnn_cell_impl import _linear as linear

from lstm import split_data as sd
from lstm.rnn import RNN

########################################################################
import pandas as pd

sensors = ('Arm', 'Belt', 'Pocket', 'Wrist')
columns = ('Time_Stamp','Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz','Activity_Label')
classes = ('Downstairs', 'Running', 'Sitting', 'Standing', 'Upstairs', 'Walking')

time_columns = ['Time_Stamp']
x_columns = ['Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz']
y_columns = ['Activity_Label']

data = {}

## CSVs are faster than XLSX
## Do NOT use Excel to save CSVs, you will lose precision
## use the script attached 
for sensor in sensors:
  fname = './data/' + sensor + '.csv'
  data[sensor] = pd.read_csv(fname, header = 0)
  data[sensor]['Activity_Label'] = data[sensor]['Activity_Label'].replace(classes, range(len(classes)))
  
data_extract = {}
for sensor in sensors:
  data_extract[sensor] = {}
  data_extract[sensor]['X'] = data[sensor][x_columns]
  data_extract[sensor]['time'] = data[sensor][time_columns]
  data_extract[sensor]['y'] = data[sensor][y_columns]

split = [.7, .1, .2]

X = data[sensors[0]][x_columns].values
y = data[sensors[0]][y_columns].values
le = preprocessing.LabelBinarizer()
y = le.fit_transform(y)
time = data[sensors[0]][time_columns].values
time = time.T[0]

## In case we use SVM:
X = StandardScaler().fit_transform(X)
## Split data
X_train, y_train, X_cv, y_cv, X_test, y_test = sd.split_data(X, y, time, split, 5000.)
#print ("SPlitData:", len(sd.split_data(X, y, time, split, 5000.)), time[-1])
# staaaahhhp()


# Network Parameters
time = 5. # We will use that much time to make a decision
sample_time = 0.02 # 20ms per sample
n_input = 9 # Data input size
n_steps = int(time / sample_time) # timesteps
n_hidden = 128 # hidden layer num of features (hyperparam)
n_classes = 6 # 

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

## Make it divisible by n_steps*batch_size
length = n_steps*batch_size - X_train.shape[0] % (n_steps*batch_size)
X_train = np.vstack((X_train, [[0]*X_train.shape[1] for _ in range(length)]))
y_train = np.vstack((y_train, [[0]*y_train.shape[1] for _ in range(length)]))
length = n_steps*batch_size - X_cv.shape[0] % (n_steps*batch_size)
X_cv = np.vstack((X_cv, [[0]*X_cv.shape[1] for _ in range(length)]))
y_cv = np.vstack((y_cv, [[0]*y_cv.shape[1] for _ in range(length)]))
length = n_steps*batch_size - X_test.shape[0] % (n_steps*batch_size)
X_test = np.vstack((X_test, [[0]*X_test.shape[1] for _ in range(length)]))
y_test = np.vstack((y_test, [[0]*y_test.shape[1] for _ in range(length)]))

########################################################################

# tf Graph input
xd = tf.placeholder("float", [None, n_steps, n_input])
yd = tf.placeholder("float", [None, n_steps, n_classes])

# Define weights
weights = {
  'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
  'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = RNN(xd, weights, biases, n_input, n_steps, n_hidden)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=yd))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(yd,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
  sess.run(init)
  step = 1

  # Keep training until reach max iterations
  while step * batch_size < training_iters:
    # batch_x, batch_y = mnist.train.next_batch(batch_size)
    start = ((step-1)*batch_size*n_steps) % X_train.shape[0]
    stop = (step*batch_size*n_steps) % X_train.shape[0]
    if start >= stop:
      step += 1
      continue
    batch_x = X_train[start:stop, :]
    batch_y = y_train[start:stop:n_steps, :]
    # print ("Batch shapes", batch_x.shape, batch_y.shape)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    # batch_y = batch_y.reshape((batch_size, n_steps, n_input))
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={xd: batch_x, yd: batch_y})

    if step % display_step == 0:
      # Calculate batch accuracy
      acc = sess.run(accuracy, feed_dict={xd: batch_x, yd: batch_y})
      # Calculate batch loss
      loss = sess.run(cost, feed_dict={xd: batch_x, yd: batch_y})
      print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))
    step += 1
    # break
  print("Optimization Finished!")

  # Calculate accuracy for 128 mnist test images
  cv_acc = 0.
  step = 1
  while step*n_steps*batch_size < y_cv.shape[0]:
    start = ((step-1)*batch_size*n_steps) % X_cv.shape[0]
    stop = (step*batch_size*n_steps) % X_cv.shape[0]
    cv_x = X_cv[start:stop,:].reshape((batch_size, n_steps, n_input))
    cv_acc += sess.run(accuracy, 
      feed_dict={xd: cv_x, yd: y_cv[start:stop:n_steps,:]})
    step+=1
  print ("CV Accuracy:", cv_acc / step)
  #print("Testing Accuracyd:", \
  #    sess.run(accuracy, feed_dict={xd: X_test[:test_len], yd: y_test[:test_len]}))
