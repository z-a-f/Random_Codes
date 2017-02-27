import numpy as np


def split_data(X, y, time, split, sep = 5000.):
  """
  Splitting the time series. X and y have to be sorted

  Args:
    X, y: input pairs
    time: timestamps of the input pairs
    split: 3 numbers that show the split ratios
    sep: time in ms to use as a threshold
  """
  assert len(split) == 3
  assert sum(split) == 1, str(sum(split))
  assert X.shape[0] == y.shape[0] == time.shape[0]

  measurement_start = [0]
  for idx, t in enumerate(time):
    if abs(t - time[measurement_start[-1]]) > sep:
      measurement_start.append(idx)
  measurement_start.append(y.shape[0])

  X_train = X[0]
  y_train = y[0]
  X_cv = X[0]
  y_cv = y[0]
  X_test = X[0]
  y_test = y[0]
  start = 0
  for stop in measurement_start[1:]:
    total = stop-start

    test_start = start
    test_stop = int(start + total*split[2])
    cv_start = test_stop
    cv_stop = int(cv_start + total*split[1])
    train_start = cv_stop

    X_test = np.vstack((X_test, X[test_start:test_stop, :]))
    y_test = np.vstack((y_test, y[test_start:test_stop, :]))
    X_cv = np.vstack((X_cv, X[cv_start:cv_stop, :]))
    y_cv = np.vstack((y_cv, y[cv_start:cv_stop, :]))
    X_train = np.vstack((X_train, X[train_start:stop, :]))
    y_train = np.vstack((y_train, y[train_start:stop, :]))

    start = stop
  return X_train[1:], y_train[1:], X_cv[1:], y_cv[1:], X_test[1:], y_test[1:]
