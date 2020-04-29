import math
import gc
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

    def __init__(self, lofar_data, labels, runs_per_classes, classes_values):
        self.lofar_data = lofar_data
        self.labels = labels
        self.runs_per_classes = runs_per_classes
        self.classes_values = classes_values
        self._compiled = False
        self.nov_cls = None

    def compile(self, test_batch, train_batch, val_batch=None, val_percentage=None, window_size=None, stride=None, convolutional_input=True):
        self.test_batch = test_batch
        self.train_batch = train_batch
        self.val_batch = val_batch
        if val_percentage == 0:
            self.val_percentage = None
        else:
            self.val_percentage = float(val_percentage)
        self.window_size = window_size
        self.stride = stride
        if self.window_size is None:
            self.mount_images=False
        else:
            self.mount_images=True
        self.convolutional_input = convolutional_input
        if self.mount_images:
            if self.convolutional_input:
                self.input_shape = (window_size, self.lofar_data.shape[-1], 1)
            else:
                self.input_shape = (window_size, self.lofar_data.shape[-1])
        else:
            self.input_shape = self.lofar_data[0].shape
        self._compiled = True
        
    def set_novelty(self, nov_cls, to_known_value):
        if nov_cls in self.classes_values.keys():
            self.nov_cls = nov_cls
        else:
            raise ValueError('The given novelty class is not on the classes parameter.')
        if callable(to_known_value):
            self.to_known_value = to_known_value
        else:
            raise ValueError('The to_known_value parameter must be callable')

    def kfold_split(self, n_splits, shuffle=False, random_state=None):
        if not self._compiled:
            raise NameError('The output parameters must be defined before calling this method. define them by using the compile method.')

        if self.mount_images:
            x_set, y_set = window_runs(self.runs_per_classes.values(), self.classes_values.values(), self.window_size, self.stride)
            sequence = LofarImgSequence
        else:
            x_set = self.lofar_data
            y_set = self.labels
            sequence = LofarSequence

        if shuffle:
            x_set, y_set = utils.shuffle_pair(x_set, y_set)

        if self.nov_cls is None:
            x_known = x_set
            y_known = y_set
        else:
            x_known = x_set[np.any(y_set != self.classes_values[self.nov_cls], axis=-1)]
            y_known = y_set[np.any(y_set != self.classes_values[self.nov_cls], axis=-1)]
            x_novelty = x_set[np.all(y_set == self.classes_values[self.nov_cls], axis=-1)]
            y_novelty = y_set[np.all(y_set == self.classes_values[self.nov_cls], axis=-1)]
        
        kfolder = KFold(n_splits)

        for fit_index, test_index in kfolder.split(x_known, y_known):

            #Collecting garbage
            gc.collect()

            if self.nov_cls is None:
                x_test = x_known[test_index]
                y_test = y_known[test_index]
                x_fit = x_known[fit_index]
                #No need to compensate
                y_fit = y_known[fit_index]
            else:
                x_test = np.concatenate((x_novelty, x_known[test_index]), axis=0)
                y_test = np.concatenate((y_novelty, y_known[test_index]), axis=0)
                x_fit = x_known[fit_index]
                #Compensating from the class taken out as novelty
                y_fit = np.apply_along_axis(self.to_known_value, axis=-1, arr=y_known[fit_index])
            
            x_test_seq = sequence(lofar_data=self.lofar_data, x_set=x_test, batch_size=self.test_batch, convolutional_input=self.convolutional_input)

            if self.val_percentage is None:
                #The percentage is treated as 0%
                train_set = sequence(lofar_data=self.lofar_data, x_set=x_fit, y_set=y_fit, batch_size=self.train_batch, 
                                     convolutional_input=self.convolutional_input)
                
                yield x_test_seq, y_test, train_set
            else:
                val_split = math.ceil(len(x_fit)*self.val_percentage)

                train_set = sequence(lofar_data=self.lofar_data, x_set=x_fit[val_split:], y_set=y_fit[val_split:], batch_size=self.train_batch,
                                     convolutional_input=self.convolutional_input)

                val_set = sequence(lofar_data=self.lofar_data, x_set=x_fit[:val_split], y_set=y_fit[:val_split], batch_size=self.val_batch,
                                   convolutional_input=self.convolutional_input)
                
                yield x_test_seq, y_test, val_set, train_set

    def leave1run_out_split(self, shuffle=True):
        if not self._compiled:
            raise NameError('The output parameters must be defined before calling this method. define them by using the compile method.')

        #Collecting garbage
        gc.collect()

        if self.nov_cls is None:
            runs_per_classes = self.runs_per_classes
            classes = self.classes
        else:
            runs_per_classes = deepcopy(self.runs_per_classes)
            novelty_runs = runs_per_classes.pop(self.nov_cls)
            classes = self.classes[self.classes != self.nov_cls]

        for class_out, run_out, test_run, train_runs_per_class in self.leave1run_out(runs_per_classes, classes):

            if self.nov_cls is None:
                num_classes_train = len(self.classes)
                train_classes=classes
                if self.mount_images:
                    x_test, y_test = window_runs([[test_run]], [class_out], self.window_size, self.stride)
                else:
                    x_test = self.lofar_data[test_run]
                    y_test = np.full(len(test_run), class_out)
            else:
                num_classes_train = len(classes)
                train_classes = np.where(classes>self.nov_cls, classes-1, classes)
                if self.mount_images:
                    x_known_test, y_known_test = window_runs([[test_run]], [class_out], self.window_size, self.stride)
                    x_novelty, y_novelty = window_runs([novelty_runs], np.full(len(novelty_runs), self.nov_cls), self.window_size, self.stride)
                else:       
                    x_known_test = self.lofar_data[test_run]
                    y_known_test = np.full(len(test_run), class_out)
                    novelty_index = np.hstack(tuple(novelty_runs))
                    x_novelty = self.lofar_data[novelty_index]
                    y_novelty = np.full(len(novelty_index), self.nov_cls)
                    del novelty_index
                x_test = np.concatenate((x_known_test, x_novelty), axis=0)
                y_test = np.concatenate((y_known_test, y_novelty), axis=0)
                del x_known_test, y_known_test, x_novelty, y_novelty
            
            if self.mount_images:
                x_fit, y_fit = window_runs(train_runs_per_class, train_classes, self.window_size, self.stride)
                sequence = LofarImgSequence
            else:
                fit_index = np.full(len(self.lofar_data), True)
                fit_index[test_run] = False
                if not self.nov_cls is None:
                    fit_index[np.hstack(tuple(novelty_runs))] = False
                x_fit = self.lofar_data[fit_index]
                y_fit = np.empty(len(self.lofar_data), dtype=type(train_classes[0]))
                for runs, train_class in zip(train_runs_per_class, train_classes):
                    y_fit[np.hstack(tuple(runs))] = train_class
                y_fit = y_fit[fit_index]
                sequence = LofarSequence
                del fit_index

            x_test_seq = sequence(lofar_data=self.lofar_data, x_set=x_test, batch_size=self.test_batch, one_hot_encode=self.one_hot_encode, 
                                          num_classes=len(self.classes), convolutional_input=self.convolutional_input)
            
            if shuffle:
                x_fit, y_fit = utils.shuffle_pair(x_fit, y_fit)

            if self.val_percentage is None:
                #The percentage is treated as 0%
                train_set = sequence(lofar_data=self.lofar_data, x_set=x_fit, y_set=y_fit, batch_size=self.train_batch, one_hot_encode=self.one_hot_encode, 
                                             num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield class_out, run_out, x_test_seq, y_test, train_set
            else:
                val_split = math.ceil(len(x_fit)*self.val_percentage)
                x_val = x_fit[:val_split]
                y_val = y_fit[:val_split]
                x_train = x_fit[val_split:]
                y_train = y_fit[val_split:]
            
                train_set = sequence(lofar_data=self.lofar_data, x_set=x_train, y_set=y_train, batch_size=self.train_batch,
                                     one_hot_encode=self.one_hot_encode, num_classes=num_classes_train, convolutional_input=self.convolutional_input)

                val_set = sequence(lofar_data=self.lofar_data, x_set=x_val, y_set=y_val, batch_size=self.val_batch, one_hot_encode=self.one_hot_encode, 
                                   num_classes=num_classes_train, convolutional_input=self.convolutional_input)
                
                yield class_out, run_out, x_test_seq, y_test, val_set, train_set

    @staticmethod
    def leave1run_out(runs_per_classes):
        """Generator that returns the range of the run left out for test and the rest for fitting
        
        Parameters:

        runs_per_classes: OrderedDict or dict(Python 3.7+)
            Dictionary with the keys as the classes_names and its respective value as an array with
            the ranges of its runs.

        Yields:

        class_out_name:
            Name of the class that was taken out
        
        run_out: int
            Index of the run that was taken out

        test_run:
            Test run

        train_runs_per_class:
            Classes and runs for training with the same format of runs_per_class
        """
    
        for class_out_name, runs in runs_per_classes.items():
            for run_out_index in range(len(runs)):
                train_runs_per_class = deepcopy(runs_per_classes)
                test_run = train_runs_per_class[class_out_name].pop(run_out_index)

                yield class_out_name, run_out_index, test_run, train_runs_per_class

    def __print__(self):
        parameters = self.__dict__
        parameters.pop('lofar_data')
        parameters['lofar_data shape'] = self.lofar_data.shape
        print('The current state of the model parameters are:')
        for key, value in parameters.items():
            print(f'{key} : {value}')
        if not self._compiled:
            print('The model has not yet been compiled, to compile its parameters for output use the compile method.')