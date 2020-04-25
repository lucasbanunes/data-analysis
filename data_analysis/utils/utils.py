import math
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence, to_categorical

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
        """Applies gradient_weights function"""
        return gradient_weights(self.y_set)

    @staticmethod
    def _numpy_array(array):
        """Tests if the array is a numpy array and returns a numpy array version if not"""
        if type(array) != np.ndarray and not array is None:
            array = np.array(array)
        return array

class _WrapperSequence(Sequence):
    """Sequence to standardize the training data to the wrapper."""

    def __init__(self, exp_pred, exp_set):
        self.exp_pred = exp_pred
        exp_set.apply(lambda x,y: (np.array(x), np.array(y)))
        self.exp_set = exp_set
        self.batch_size = len(exp_set[0][0])
        
    def __len__(self):
        """Returns the number of bacthes""" 
        return len(self.exp_set)

    def __getitem__(self, index):
        batch_x = self.exp_pred[index*self.batch_size:(index+1)*self.batch_size]
        batch_y = self.exp_set[index][1]

        return batch_x, batch_y

    def gradient_weights(self):
        return self.exp_set.gradient_weights()

def gradient_weights(y):
    """Returns a dict with a key for each class and each value a weight for that class.
    The weights are inversely propotional to the number of that class occurence:
    weight = min occurences of a class/occurences of the current class
    The dict is compatible with class_weight from keras Sequential model fit

    Parameters:

    y: numpy array
        Correct labels of a set
    """
    ndim = len(y.shape)
    if ndim == 1:
        pass
    elif ndim ==2:
        y = np.argmax(y, axis=1)
    else:
        raise ValueError(f'Could not retrive gradient weights from array with dimension {ndim}')
    
    classes, occurences = np.unique(y, axis=0, return_counts=True)
    min_occurence = min(occurences)
    return {int(class_): float(min_occurence / occurence) for class_, occurence in zip(classes, occurences)}

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

def cast_to_python(var):
    """Casts a variable with other packages types to its respective python type if possible and returns it"""
    typo = type(var)
    if (typo == np.float32) or (typo == np.float64) or (typo == np.float16):
        return float(var)
    elif (typo == np.int16) or (typo == np.int32) or (typo == np.int64):
        return int(var)
    elif typo == str:
        return var
    elif typo == int:
        return var
    elif typo == float:
        return var
    elif typo == bool:
        return var
    elif typo == type(None):
        return None
    else:
        raise ValueError(f'{typo} casting is not supported')

def cast_dict_to_python(dictionary):
    """Casts a dict with key and values with other packages types to its respective python type if possible and returns it."""
    d = dict()
    for key, value in dictionary.items():
        if type(value) == list or type(value) == np.ndarray:
            d[key] = [cast_to_python(var) for var in value]
        elif type(value) == dict:
            d[key] = cast_dict_to_python(value)
        elif type(value) == tuple:
            d[key] = (cast_to_python(var) for var in value)
        else:
            d[key] = cast_to_python(value)

    return d

def to_sparse_tanh(y, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y))
    sparse_tanh = np.full(shape=(len(y), num_classes), fill_value=-1, dtype=np.int32)
    for class_, event in zip(y,sparse_tanh):
        event[class_] = 1
    return sparse_tanh

def around(num):
    decimals=0
    i = 1
    while True:
        if num>i:
            return round(num, decimals), decimals
        else:
            decimals += 1
            i = i*0.1

class NumericalIntegration():
    def __init__(self):
        pass

    @staticmethod
    def trapezoid_rule(x, y):
        area = 0
        for k in range(1, len(x)):
            dx = x[k]-x[k-1]
            area += (y[k]+y[k-1]) * dx / 2

        return area