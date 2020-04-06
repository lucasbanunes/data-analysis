import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence

class DataSequence(Sequence):
    """Base class for Data Sequences inheriting from keras.utils.Sequence class

    Attributes:
    
    x_set: array like
        x set of the data
    
    y_set: array like
        y set from the data
    """
    def __init__(self, x_set, y_set=None):
        self.x_set = self._numpy_array(x_set)
        self.y_set = self._numpy_array(y_set)

    def apply(self, function, args=None):
        """Applies a function to the x and y sets.
        The function must recieve two parameters at minimum the x and y sets, other ones must be passed
        to the args parameter as a dict mapping.

        Parameters:
        
        function:
            Python function to be applied to the sets
        
        args: dict
            Python dictionary that maps the other parameters of function

        Return:
        x_set, y_set: numpy array, numpy array
            Arrays returned from the function
        """
        if args is None:
            self.x_set, self.y_set = function(self.x_set, self.y_set)
        else:
            args['x_set'] = self.x_set
            args['y_set'] = self.y_set
            x_set, y_set = function(**args)
            self.x_set = self._numpy_array(x_set)
            self.y_set = self._numpy_array(y_set)
        return self.x_set, self.y_set

    def gradient_weights(self):
        """Returns a dict with a key for each class and each value a weight for that class.
        The weights are inversely propotional to the number of that class occurence:
        weight = min occurences of a class/occurences of the current class
        The dict is compatible with class_weight from keras Sequential model fit
        """
        classes, occurences = np.unique(self.y_set, axis=0, return_counts=True)
        min_occurence = min(occurences)
        return {int(class_): float(min_occurence / occurence) for class_, occurence in zip(classes, occurences)}

    @staticmethod
    def _numpy_array(array):
        """Tests if the array is a numpy array and returns a numpy array version if not"""
        if type(array) != np.ndarray and not array is None:
            array = np.array(array)
        return array

def class_name(obj):
    """Returns the name of the obj's class"""
    return obj.__class__.__name__

def frame_from_history(dict_):
    """Builds a pandas.DataFrame from the keras.callbacks.callbacks.History.history
    attribute"""
    
    frame = pd.DataFrame.from_dict(dict_)
    for value in dict_.values():
        epochs = len(value)
        break
    frame.index = pd.Index(range(epochs), name='epoch')

    return frame

def shuffle_pair(x, y):
    """Shuffles a pair of data"""
    pair = list(zip(x, y))
    np.random.shuffle(pair)
    x=list()
    y=list()
    for x_value, y_value in pair:
        x.append(x_value)
        y.append(y_value)
    
    return np.array(x), np.array(y)

def reshape_conv_input(data):
    """Returns an numpy.ndarray reshaped as an input for a convolutional layer from keras

    Parameters:

    data: numpy.ndarray
        Data to be reshaped
    """
    
    shape = list(data.shape)
    shape.append(1)
    data.reshape(tuple(shape))
    return data.reshape(tuple(shape))