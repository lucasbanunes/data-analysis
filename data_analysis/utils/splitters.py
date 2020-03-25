import math
from copy import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence, to_categorical

def train_val_split(x_set, y_set, val_percentage=0.0, shuffle=True):
    """Splits a data set into train and validation subsets for fitting
    a model

    Parameters:

    x_set: numpy.ndarray
        X set of the data

    y_set: numpy..ndarray
        Y set of the data

    val_percentage: float
        Percentage of the data, from 0 to 1,  that will be used as
        validation data
    
    shuffle: boolean
        If true shuffles the set
    
    Returns:

    x_set_train:numpy.ndarray
    y_set_train:numpy.ndarray
    x_set_val:numpy.ndarray
    y_set_val:numpy.ndarray
    """

    if shuffle:
        x_y_set = list(zip(x_set, y_set))
        np.random.shuffle(x_y_set)
        x_set = list()
        y_set = list()
        for x, y in x_y_set:
            x_set.append(x)
            y_set.append(y)
        x_set = np.array(x_set)
        y_set = np.array(y_set)

    split = math.ceil(len(x_set)*val_percentage)
    x_set_train = x_set[:split]
    x_set_val = x_set[split:]
    y_set_train = y_set[:split]
    y_set_val = y_set[split:]

    return x_set_train, y_set_train, x_set_val, y_set_val

def leave_one_run_out(classes_runs):
    """Generator that returns the range of the run left out for test and the rest for fitting
    
    Parameters:

    classes_runs: 2-d arraylike
        Array with first dimension being the number of classes
        each class has a 2-d tuple. The first element of the tuple is
        a array with the ranges of the runs and the second element the class.

        Example:
        class_1_runs = [range(0,100), range(100,200)]
        class_3_runs =  [range(630, 730)]
        classes_runs = [class_1_runs, class_3_runs]
        classes = [1,3]
        classes_runs = list(zip(classes_runs, classes))

    Yields:

    test_run: numpy.ndarray with shape (1,1)
        Array with the test run
    
    fit_classes_runs: list of lists
        List with the rest of the runs separated by class
    """
    
    for class_ in range(len(classes_runs)):
        for run in range(len(classes_runs[class_])):
            test_run = np.array(classes_runs[class_][run]).reshape(1,1)
            train_classes_runs = copy(classes_runs)
            train_classes_runs[class_].pop(run)

            yield test_run, train_classes_runs
