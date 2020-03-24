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