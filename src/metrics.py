import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(((y_true * y_pred) >= 0)*np.abs(y_true))/np.sum(np.abs(y_true))


