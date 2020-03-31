import math
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import KFold
import data_analysis.utils.utils as utils
from data_analysis.utils.lofar_operators import LofarImgSequence, LofarSequence, window_runs

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

class LofarSplitter():

    def __init__(self, lofar_data, classes_runs, classes, window_size, stride):
        self.lofar_data = lofar_data
        self.classes_runs = classes_runs
        self.window_size = window_size
        self.stride = stride
        self.classes = classes
        self.nov_cls = None

    def compile(self, test_batch, train_batch, val_batch=None, val_percentage=None, one_hot_encode=False, mount_images=False, convolutional_input=True):
        self.test_batch = test_batch
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.one_hot_encode = one_hot_encode
        self.convolutional_input = convolutional_input
        self.mount_images = mount_images
        if val_percentage == 0:
            self.val_percentage = None
        else:
            self.val_percentage = float(val_percentage)

    def set_novelty(self, nov_cls):
        if nov_cls in self.classes:
            self.nov_cls = nov_cls
        else:
            raise ValueError('The given novelty class is not on the classes parameter.')

    def kfold_split(self, n_splits, shuffle=False, random_state=None):
        if not self._compiled():
            raise NameError('The output parameters must be defined before calling this method. define them by using the compile method.')

        sequence = self._get_sequence()
        x_set, y_set = window_runs(list(zip(self.classes_runs, self.classes)), self.window_size, self.stride)

        if not self.nov_cls is None:
            x_known = x_set[y_set != self.nov_cls]
            y_known = y_set[y_set != self.nov_cls]
            x_novelty = x_set[y_set == self.nov_cls]
            y_novelty = y_set[y_set == self.nov_cls]
        else:
            x_known = x_set
            y_known = y_set

        kfolder = KFold(n_splits, shuffle, random_state)

        for fit_index, test_index in kfolder.split(x_known, y_known):

            if not self.nov_cls is None:
                x_test = np.concatenate((x_novelty, x_known[test_index]), axis=0)
                y_test = np.concatenate((y_novelty, y_known[test_index]), axis=0)
                x_fit = x_known[fit_index]
                #Compensating from the class taken out as novelty
                y_fit = np.where(y_known[fit_index]>self.nov_cls, y_known[fit_index]-1, y_known[fit_index])
                num_classes_train = len(self.classes)-1
            else:
                x_test = x_known[test_index]
                y_test = y_known[test_index]
                x_fit = x_known[fit_index]
                #No need to compensate
                y_fit = y_known[fit_index]
                num_classes_train = len(self.classes)
            
            x_test_seq = sequence(self.lofar_data, x_test, batch_size=self.test_batch, one_hot_encode=self.one_hot_encode, 
                                          num_classes=len(self.classes), convolutional_input=self.convolutional_input)

            if self.val_percentage is None:
                #The percentage is treated as 0%
                train_set = sequence(self.lofar_data, x_fit, y_fit, self.train_batch, self.one_hot_encode, 
                                             num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield x_test_seq, y_test, train_set
            else:
                val_split = math.ceil(len(x_fit)*self.val_percentage)
                x_val = x_fit[:val_split]
                y_val = y_fit[:val_split]
                x_train = x_fit[val_split:]
                y_train = y_fit[val_split:]
            
                train_set = sequence(self.lofar_data, x_train, y_train, self.train_batch, self.one_hot_encode, 
                                            num_classes=num_classes_train, convolutional_input=self.convolutional_input)

                val_set = sequence(self.lofar_data, x_val, y_val, self.val_batch, self.one_hot_encode, 
                                           num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield x_test_seq, y_test, val_set, train_set

    def leave1run_out_split(self):
        if not self._compiled():
            raise NameError('The output parameters must be defined before calling this method. define them by using the compile method.')

        sequence = self._get_sequence()

        for class_out, run_out, test_run, train_classes_runs in self.leave1run_out(self.classes_runs):
            x_set, y_set = window_runs(list(zip(self.classes_runs, self.classes)), self.window_size, self.stride)
            x_test, y_test = window_runs([(test_run, class_out)], self.window_size, self.stride)

            if not self.nov_cls is None:
                x_fit = x_set[y_set != self.nov_cls]
                #Compensating from the class taken out as novelty
                y_fit = np.where(y_set[y_set != self.nov_cls]>self.nov_cls, y_set[y_set != self.nov_cls] - 1, y_set[y_set != self.nov_cls])
                x_novelty = x_set[y_set == self.nov_cls]
                y_novelty = y_set[y_set == self.nov_cls]
                num_classes_train = len(self.classes)-1
            else:
                #No need to compensate
                x_fit = x_set
                y_fit = y_set
                num_classes_train = len(self.classes)

            x_test_seq = sequence(self.lofar_data, x_test, batch_size=self.test_batch, one_hot_encode=self.one_hot_encode, 
                                          num_classes=len(self.classes), convolutional_input=self.convolutional_input)

            if self.val_percentage is None:
                #The percentage is treated as 0%
                train_set = sequence(self.lofar_data, x_fit, y_fit, self.train_batch, self.one_hot_encode, 
                                             num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield class_out, run_out, x_test_seq, y_test, train_set
            else:
                val_split = math.ceil(len(x_fit)*self.val_percentage)
                x_val = x_fit[:val_split]
                y_val = y_fit[:val_split]
                x_train = x_fit[val_split:]
                y_train = y_fit[val_split:]
            
                train_set = sequence(self.lofar_data, x_train, y_train, self.train_batch, self.one_hot_encode, 
                                            num_classes=num_classes_train, convolutional_input=self.convolutional_input)

                val_set = sequence(self.lofar_data, x_val, y_val, self.val_batch, self.one_hot_encode, 
                                           num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield class_out, run_out, x_test_seq, y_test, val_set, train_set

    @staticmethod
    def leave1run_out(classes_runs):
        """Generator that returns the range of the run left out for test and the rest for fitting
        
        Parameters:

        classes_runs: 2-d arraylike
            Array with first dimension being the number of classes and second being the runs range.

        Yields:

        test_run: numpy.ndarray with shape (1,1)
            Array with the test run
        
        fit_classes_runs: list of lists
            List with the rest of the runs separated by class
        """
    
        for class_out in range(len(classes_runs)):
            for run_out in range(len(classes_runs[class_out])):
                train_classes_runs = deepcopy(classes_runs)
                test_run = [train_classes_runs[class_out].pop(run_out)]

                yield class_out, run_out, test_run, train_classes_runs
    
    def _compiled(self):
        try:
            self.test_batch
            self.train_batch
            self.val_batch
            self.val_percentage
            self.convolutional_input
        except NameError:
            return False
        return True

    def __print__(self):
        parameters = self.__dict__
        parameters.pop('lofar_data')
        parameters['lofar_data shape'] = self.lofar_data.shape
        print('The current state of the model parameters are:')
        for key, value in parameters.items():
            print(f'{key} : {value}')
        if not self._compiled():
            print('The model has not yet been compiled, to compile its parameters for output use the compile method.')

    def _get_sequence(self):
        if self.mount_images:
            return LofarImgSequence
        else:
            return LofarSequence