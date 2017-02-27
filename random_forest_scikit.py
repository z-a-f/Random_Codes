#!/usr/bin/env python

# ## Make sure dependencies exist
# import imp
# requirements = (
#   # For math and stuff
#   'pandas', 
#   'numpy',
#   # To store and load the models
#   'pickle',
#   # For ML stuff
#   'sklearn', 
#   'tensorflow',
# )


# for req in requirements:
#   try:
#     imp.find_module(req)
#   except ImportError:
#     print "Cannot find '{req}' module! The codes might not work! How annoying!!!".format(req=req)

import pandas as pd
import numpy as np

sensors = ('Arm', 'Belt', 'Pocket', 'Wrist')
columns = ('Time_Stamp','Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz','Activity_Label')
classes = ('Downstairs', 'Running', 'Sitting', 'Standing', 'Upstairs', 'Walking')
split = (.7, .1, .2)

# sensors = ['Arm']
data = {}
for sensor in sensors:
  print 'Loading ' + sensor + '...'
  fname = './data/' + sensor + '.csv'
  data[sensor] = pd.read_csv(fname, header = 0)
  data[sensor]['Activity_Label'] = data[sensor]['Activity_Label'].replace(classes, range(len(classes)))

### Attempt 1: Single sample classifiers -- violates Shannon-Nyquist Sampling!
# For this attempt the time stamps don't play a role. In fact this feature creates a problem: it would put a lot of emphasis on timestamps, as we know that the input data is periodic. We will also ignore the possible correlation between different sensors

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

## Exclude the timestamps, and combine the sensor data
time_columns = ['Time_Stamp']
x_columns = ['Ax','Ay','Az','Gx','Gy','Gz','Mx','My','Mz']
y_columns = ['Activity_Label']

# time = data[sensors[0]][time_columns].values
X = data[sensors[0]][x_columns].values
y = data[sensors[0]][y_columns].values

for sensor in sensors[1:]:
  X = np.append(X, data[sensor][x_columns].values, axis = 0)
  y = np.append(y, data[sensor][y_columns].values, axis = 0)

y = y.T[0]

## In case we use SVM:
X = StandardScaler().fit_transform(X)

## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split[-1], random_state=42)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=split[1] / (split[0]+split[1]), random_state=42)

## Try a simple RF classifier
import pickle as pkl
depths = [5, 7, 10, None]
n_estimators = [15, 20, 30, 50]

RF = []
train_mat = []
cv_mat = []
test_mat = []
for n in n_estimators:
  train_row = []
  cv_row = []
  test_row = []
  
  for depth in depths:
    print "Training", n, depth
    RF.append(RandomForestClassifier(n_estimators = n, max_depth = depth))
    RF[-1].fit(X_train, y_train)
    train_row.append(RF[-1].score(X_train, y_train))
    cv_row.append(RF[-1].score(X_cv, y_cv))
    test_row.append(RF[-1].score(X_test, y_test))

  train_mat.append(np.copy(train_row))
  cv_mat.append(np.copy(cv_row))
  test_mat.append(np.copy(test_row))

RES = [
  pd.DataFrame(train_mat, columns = depths, index = n_estimators),
  pd.DataFrame(cv_mat, columns = depths, index = n_estimators),
  pd.DataFrame(test_mat, columns = depths, index = n_estimators),
]

print "Saving a model"
with open('./models/rf_all.model', 'w') as f:
  pkl.dump(RF, f)
with open('./models/rf_all.results', 'w') as f:
  pkl.dump(RES, f)
