import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence, to_categorical
from data_analysis.utils.utils import reshape_conv_input, shuffle_pair, DataSequence

def lofar2img(lofar_data, windows):
    """Function that takes the windows and the lofar_data and mount the
    images

    Parameters:

    lofar_data: numpy.ndarray
        Full lofar spectrogram where x is frequency and y is time

    self.windows: array like
        Array with the range of the windows to be used to mount the images

    Returns:

    images: numpy.ndarray
        Array with the images mountes from the windows
    """

    images = list()
    for window in windows:
        images.append(lofar_data[window])
    
    return np.array(images)

def window_runs(runs_per_class, classes, window_size, stride):
    """Gets the windows ranges from multiple classes with multiple runs
    from a lofar spectrogram using a sliding window

    Parameters:

    runs_per_classes: 2D array-like
        Array with first dimension being the number of classes and second being the runs from that class.

    classes: 1D array-like
        Respective class for each run fom runs_per class

    window_size: int
        Size of the window
    
    stride: int
        Step that the silding window takes from a window to another

    Returns:

    windows: numpy.ndarray
        Array with the windows ranges
    
    win_labels: numpy.ndarray
        respective windows labels
    """

    windows = list()
    win_labels = list()
    for runs, run_label in zip(runs_per_class, classes):
        for run in runs:
            run_start = run[0]
            run_end = run[-1]
            for win_start in range(run_start, run_end+1, stride):
                win_end = win_start+window_size
                if win_end>run_end:     #The window gets out of the run range
                    break
                windows.append(range(win_start, win_end))
                win_labels.append(run_label)
    
    return np.array(windows), np.array(win_labels)

class LofarImgSequence(DataSequence):
    """Child class of keras.utils.Sequence that generates lofar images
    to get the data on demand for neural network fitting

    Attributes:

    self.lofar_data: numpy.ndarray
        Full lofar spectrogram where x is frequency and y is time
    
    self.x_set: array like
        Array with the range of the windows to be used to mount the images
    
    self.y_set: 1-d array like
        Array with the respective window label
    
    self.batch_size: int
        Size of the batch. Default is 32.
    
    self.one_hot_encode: boolean
        If true the y set is one hot encoded. Default is False

    self.num_classes: int
        Number of casses to be considered for the one hot encoding

    self.convolutional_input: boolean
        Keras convolutional layers work with an extra dim for images in this case.
        If true the window instad of being (shape[0], shape[1]) it is
        (shape[0], shape[1], 1)
    """

    def __init__(self, lofar_data, x_set, y_set=None, batch_size=32, one_hot_encode=False, num_classes = None, convolutional_input=True, **kwargs):
        super().__init__(x_set, y_set)
        self.lofar_data = lofar_data
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        self.convolutional_input = convolutional_input
        if type(num_classes) == type(None):
            self.num_classes = len(np.unique(y_set))
        else:
            self.num_classes = num_classes

    def __len__(self):
        """Returns the number of bacthes""" 
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        """Gets batch at position index

        Parameters:

        index: int
            Position of the batch in the Sequence

        Returns:
            A batch
        """

        batch_x = self.x_set[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = lofar2img(self.lofar_data, batch_x)   #the batch is now the images and not the windows

        if self.convolutional_input:
            batch_x = reshape_conv_input(batch_x)

        if not self.y_set is None:
            batch_y = np.array(self.y_set[index*self.batch_size:(index+1)*self.batch_size])
            if self.one_hot_encode:
                batch_y = to_categorical(batch_y, self.num_classes)
            
            return batch_x, batch_y
        else:
            return batch_x

class LofarSequence(DataSequence):
    """Child class from keras.utils.Sequence that returns each line of a lofargram
    
    Attributes:

    self.x_set: numpy.ndarray
        X set of the dataset

    self.y_set: numpy.ndarray
        Y set of the dataset

    self.batch_size: int
        Size of the batch
    
    self.one_hot_encode: boolean
        If true the y set is one hot encoded. Default is False

    self.num_classes: int
        Number of casses to be considered for the one hot encoding
    """

    def __init__(self, x_set, y_set=None, batch_size=32, one_hot_encode=False, num_classes=None, **kwargs):
        super().__init__(x_set, y_set)
        self.batch_size = batch_size
        self.one_hot_encode = one_hot_encode
        if type(num_classes) == type(None):
            self.num_classes = len(np.unique(y_set))
        else:
            self.num_classes = num_classes

    def __len__(self):
        """Returns the number of bacthes""" 
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, index):
        """Gets batch at position index

        Parameters:

        index: int
            Position of the batch in the Sequence

        Returns:
            A batch
        """
        batch_x = self.x_set[index*self.batch_size:(index+1)*self.batch_size]
        if self.y_set is None:
            return batch_x
        else:
            batch_y = self.y_set[index*self.batch_size:(index+1)*self.batch_size]
            if self.one_hot_encode:
                batch_y = to_categorical(batch_y, self.num_classes)
            return batch_x, batch_y