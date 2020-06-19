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
            x_set, y_set= function(self.x_set, self.y_set)
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

    def input_shape(self):
        input_shape = self[0][0].shape[1:]
        return input_shape

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

    def input_shape(self):
        input_shape = self[0][0].shape[1:]
        return input_shape

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
    """Shuffles a pair of data. Thre returned data will be a ndarray"""
    if not ((type(x) is np.ndarray) and (type(y) is np.ndarray)):
        x = np.array(x)
        y = np.array(y)

    index = np.arange(len(x))
    np.random.shuffle(index)

    return x[index], y[index]

def sort_pair(x, y):
    """Sorts a pair using x as source for sorting."""
    if not ((type(x) is np.ndarray) and (type(y) is np.ndarray)):
        x = np.array(x)
        y = np.array(y)

    sorted_index = x.argsort(axis=0)

    return x[sorted_index], y[sorted_index]

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
    """Casts a variable from other packages types to a respective python type if possible and returns the new one"""
    typo = type(var)
    default_types = [int, float, str, bool]
    if (typo in default_types) or (var is None):
        return var
    elif (typo == np.float32) or (typo == np.float64) or (typo == np.float16):
        return float(var)
    elif (typo == np.int16) or (typo == np.int32) or (typo == np.int64):
        return int(var)
    elif (typo == list) or (typo == np.ndarray):
        return [cast_to_python(value) for value in var]
    elif typo == tuple:
        return tuple([cast_to_python(value) for value in var])
    elif typo == dict:
        return {cast_to_python(key): cast_to_python(value) for key, value in var.items()}
    else:
        raise ValueError(f'{typo} casting is not supported')

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
    while decimals<10:
        if num>i:
            return round(num, decimals), decimals
        decimals += 1
        i = i*0.1
    return 0.0, decimals

class LoopRange():
    def __init__(self, start, stop, step=None, initial_value=None, num_samples=None):
        self.start = start
        self.stop = stop
        self.generated = 0
        if initial_value is None:
            self.current=start
        else:
            self.current=initial_value
        if step is None:
            self.step = 1
        else:
            self.step = step
        if num_samples is None:
            self.num_samples=abs(start-stop)-1
        else:
            self.num_samples = num_samples

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.generated < self.num_samples:
            if self.current == self.stop:
                self.current = self.start
            num = self.current
            self.current += self.step
            self.generated += 1
            return num
        raise StopIteration

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

def second_max(iterable):
    max_item = max(iterable)
    if iterable[0] == max_item:
        second = iterable[1]
    else:
        second = iterable[0]
    for item in iterable:
        if item > second and item < max_item:
            second = item
    return second