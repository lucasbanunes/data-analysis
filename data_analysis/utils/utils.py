import pandas as pd
import numpy as np

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