import math
import gc
from copy import deepcopy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import KFold
from data_analysis.utils import math_utils
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
    x_set_val = x_set[:split]
    x_set_train = x_set[split:]
    y_set_val = y_set[:split]
    y_set_train = y_set[split:]

    return x_set_train, y_set_train, x_set_val, y_set_val


def most_even_split(index_range, n_splits):
    """Slices a index array with the given number of splits in the most even manner possible
    Example:
    The range(13) object can be seen as
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    divided in 5 splits could happen in any number of ways but this method splits it the most 
    even way possible that is two splits with 2 events and 3 wplits with 3 elements.

    Parameters:

    index_range: 1D array-like
        A 1D iterable with its elements being ints that simbolizes indexes
    
    n_splits: int
        Number of splits to be done

    Returns:

    splits: list
        A list with each item being a split array with the indexes of that split
    """
    events = len(index_range)
    index_start = index_range[0]
    index_end = index_range[-1] + 1
    events_per_split = int(events/n_splits)
    remainders = events%n_splits
    if remainders == 0:     #The array is splitted evenly
        splits =  [np.arange(split_start, split_start+events_per_split) for split_start in range(index_start, index_end, events_per_split)]
    else:
        #Mouting the uneven splits
        splits = list()
        start = index_start
        events_to_add = math_utils.euclidean_algorithm(remainders, n_splits)
        for _ in range(n_splits):
            if remainders>0:
                to_add = events_per_split + events_to_add
                remainders -= events_to_add
            elif remainders == 0:
                to_add = events_per_split
            else:
                raise ValueError(f'Reached a negative remainder: {remainders}')
            splits.append(np.arange(start, start+to_add))
            start += to_add
    return splits

class LofarSplitter():

    def __init__(self, lofar_data, labels, runs_per_classes, classes_values_map):
        self.lofar_data = lofar_data
        self.labels = labels
        self.runs_per_classes = runs_per_classes
        self.classes_values_map = classes_values_map
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
        if nov_cls in self.classes_values_map.keys():
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
            sequence = LofarImgSequence
            x_set, y_set, runs_range = window_runs(self.runs_per_classes.values(), self.classes_values_map.values(), self.window_size, self.stride)
            if self.nov_cls is None:
                x_known, y_known, known_range = x_set, y_set, runs_range
            else:
                x_known, y_known, known_range, x_novelty, y_novelty, novelty_range = self._remove_novelty(x_set, y_set, 
                                                                                            runs_range, self.classes_values_map[self.nov_cls])
        else:
            sequence = LofarSequence
            runs_range = np.concatenate(tuple(self.runs_per_classes.values()))
            if self.nov_cls is None:
                x_known = self.lofar_data
                y_known = self.labels
                known_range = runs_range
            else:
                x_known, y_known, known_range, x_novelty, y_novelty, novelty_range = self._remove_novelty(self.lofar_data, self.labels, 
                                                                            runs_range, self.classes_values_map[self.nov_cls])
        
        for test_index, val_index, train_index in self.run_kfold_split(known_range, n_splits, shuffle):

            #Collecting garbage
            gc.collect()

            x_train = x_known[train_index]
            y_train = y_known[train_index]
            x_val = x_known[val_index]
            y_val = y_known[val_index]

            if self.nov_cls is None:
                x_test = x_known[test_index]
                y_test = y_known[test_index]                
            else:
                x_test = np.concatenate((x_novelty, x_known[test_index]), axis=0)
                y_test = np.concatenate((y_novelty, y_known[test_index]), axis=0)
                #Compensating from the class taken out as novelty
                y_train = np.apply_along_axis(self.to_known_value, axis=-1, arr=y_train)
                y_val = np.apply_along_axis(self.to_known_value, axis=-1, arr=y_val)
            
            test_seq = sequence(lofar_data=self.lofar_data, x_set=x_test, y_set=y_test, batch_size=self.test_batch, 
                                convolutional_input=self.convolutional_input)

            val_seq = sequence(lofar_data=self.lofar_data, x_set=x_val, y_set=y_val, batch_size=self.val_batch, 
                                convolutional_input=self.convolutional_input)

            train_seq = sequence(lofar_data=self.lofar_data, x_set=x_train, y_set=y_train, batch_size=self.train_batch, 
                                convolutional_input=self.convolutional_input)
            
            yield test_seq, y_test, val_seq, train_seq

    def leave1run_out_split(self, shuffle=True):
        if not self._compiled:
            raise NameError('The output parameters must be defined before calling this method. define them by using the compile method.')

        #Collecting garbage
        gc.collect()

        if self.nov_cls is None:
            runs_per_classes = self.runs_per_classes
            classes_values_map = self.classes_values_map
            train_classes_values_map = self.classes_values_map
        else:
            runs_per_classes = deepcopy(self.runs_per_classes)
            novelty_runs = runs_per_classes.pop(self.nov_cls)
            train_classes_values_map = deepcopy(self.classes_values_map)
            train_classes_values_map.pop(self.nov_cls)
           

        for class_out_name, run_out_index, test_run, train_runs_per_class in self.leave1run_out(runs_per_classes):

            if self.nov_cls is None:
                if self.mount_images:
                    x_test, y_test, _ = window_runs([[test_run]], [self.classes_values_map[class_out_name]], self.window_size, self.stride)
                else:
                    x_test = self.lofar_data[test_run]
                    y_test = self.labels[test_run]
            else:
                if self.mount_images:
                    x_known_test, y_known_test, _ = window_runs([[test_run]], [self.classes_values_map[class_out_name]], self.window_size, self.stride)
                    x_novelty, y_novelty, _ = window_runs([novelty_runs], [self.classes_values_map[self.nov_cls]], self.window_size, self.stride)
                else:       
                    x_known_test = self.lofar_data[test_run]
                    y_known_test = self.labels[test_run]
                    novelty_index = np.hstack(tuple(novelty_runs))
                    x_novelty = self.lofar_data[novelty_index]
                    y_novelty = self.labels[novelty_index]
                    del novelty_index
                
                x_test = np.concatenate((x_known_test, x_novelty), axis=0)
                y_test = np.concatenate((y_known_test, y_novelty), axis=0)
                del x_known_test, y_known_test, x_novelty, y_novelty
            
            if self.mount_images:
                x_fit, y_fit, _ = window_runs(train_runs_per_class.values(), train_classes_values_map.values(), self.window_size, self.stride)
                sequence = LofarImgSequence
            else:
                fit_index = np.hstack(np.hstack(train_runs_per_class.values()))
                x_fit = self.lofar_data[fit_index]
                y_fit = self.labels[fit_index]
                sequence - LofarSequence
                del fit_index

            if not self.nov_cls is None:    #Need to compensate the training and val data for missing class
                print(y_fit)
                y_fit = np.apply_along_axis(self.to_known_value, axis=-1, arr=y_fit)

            x_test_seq = sequence(lofar_data=self.lofar_data, x_set=x_test, y_set=y_test, batch_size=self.test_batch, 
                                  convolutional_input=self.convolutional_input)
            
            if shuffle:
                x_fit, y_fit = utils.shuffle_pair(x_fit, y_fit)

            if self.val_percentage is None:
                #The percentage is treated as 0%
                train_set = sequence(lofar_data=self.lofar_data, x_set=x_fit, y_set=y_fit, batch_size=self.train_batch,
                                     convolutional_input=self.convolutional_input)
                
                yield class_out_name, run_out_index, x_test_seq, y_test, train_set
            else:
                val_split = math.ceil(len(x_fit)*self.val_percentage)
                x_val = x_fit[:val_split]
                y_val = y_fit[:val_split]
                x_train = x_fit[val_split:]
                y_train = y_fit[val_split:]
            
                train_set = sequence(lofar_data=self.lofar_data, x_set=x_train, y_set=y_train, batch_size=self.train_batch,
                                     convolutional_input=self.convolutional_input)

                val_set = sequence(lofar_data=self.lofar_data, x_set=x_val, y_set=y_val, batch_size=self.val_batch,
                                   convolutional_input=self.convolutional_input)
                
                yield class_out_name, run_out_index, x_test_seq, y_test, val_set, train_set

    @staticmethod
    def run_kfold_split(runs_range, n_splits, shuffle=False):

        runs_splitted = list()

        for run_range in runs_range:
            splits = most_even_split(run_range,n_splits)
            if shuffle:
                np.random.shuffle(splits)
            runs_splitted.append(splits)
        
        runs_splitted = np.array(runs_splitted).T

        for test_split, val_split in zip(range(n_splits),utils.LoopRange(0,n_splits, num_samples=n_splits, initial_value=1)):
            splits_arr = np.arange(n_splits)
            train_splits = np.all(np.stack((splits_arr != test_split,  splits_arr!= val_split),axis=0), axis=0)
            test_index = np.hstack(runs_splitted[test_split])
            val_index = np.hstack(runs_splitted[val_split])
            train_index = np.hstack(np.hstack(runs_splitted[train_splits]))

            if shuffle:
                np.random.shuffle(test_index)
                np.random.shuffle(val_index)
                np.random.shuffle(train_index)

            yield test_index, val_index, train_index

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
    
    @staticmethod
    def _rearrange_runs(runs_range):
        new_runs_range = list()
        current_range_start = 0
        for run_range in runs_range:
            if run_range[0] == current_range_start:
                new_runs_range.append(run_range)
            else:
                new_range = range(current_range_start, current_range_start+len(run_range))
                new_runs_range.append(new_range)
            current_range_start += len(run_range)
        return new_runs_range

    def _remove_novelty(self, x_set, y_set, runs_range, novelty_label):
        range_labels = np.array([y_set[run_range][0] for run_range in runs_range])
        novelty_index = np.all(y_set == novelty_label, axis=-1)
        known_index = np.logical_not(novelty_index)
        x_known, y_known, known_range = x_set[known_index], y_set[known_index], self._rearrange_runs(runs_range[np.any(range_labels != novelty_label, axis=-1)])
        x_novelty, y_novelty, novelty_range = x_set[novelty_index], y_set[novelty_index], self._rearrange_runs(runs_range[np.all(range_labels != novelty_label, axis=-1)])

        return x_known, y_known, known_range, x_novelty, y_novelty, novelty_range

    def __print__(self):
        parameters = self.__dict__
        parameters.pop('lofar_data')
        parameters['lofar_data shape'] = self.lofar_data.shape
        print('The current state of the model parameters are:')
        for key, value in parameters.items():
            print(f'{key} : {value}')
        if not self._compiled:
            print('The model has not yet been compiled, to compile its parameters for output use the compile method.')