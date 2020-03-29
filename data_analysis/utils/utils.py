import pandas as pd

def class_name(obj):
    """Returns the name of the obj's class"""
    return obj.__class__.__name__

def frame_from_history(self, dict_):
    """Builds a pandas.DataFrame from the keras.callbacks.callbacks.History.history
    attribute"""
    
    frame = pd.DataFrame.from_dict(dict_)
    for key in dict_.values():
        epochs = len(key)
        break
    frame.index = pd.Index(range(epochs), name='epoch')

    return frame

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