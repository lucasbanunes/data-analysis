import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from data_analysis.functions.utils import class_name


class SequentialModel():
    """ Implements the Keras package Sequential Model with fitting with mutiple inits to avoid bad starting"""

    def __init__(self, **layers):

        #Attributes
        self.model = Sequential()
        self.layers_config = OrderedDict()
        self.optimizer_config = OrderedDict()
        self.loss_config = OrderedDict()
        self.trained = False

        if layers:
            self.model = Sequential()
            for layer_name, layer_parameters in layers.items():
                layer = getattr(keras.layers, layer_name)(**layer_parameters)
                self.layers_config[layer_name] = layer.get_config()
            
        gc.collect()    #Collecting the layers instances
    
    def compile(self, optimizer, 
                loss=None, 
                metrics=['accuracy'], 
                loss_weights=None, 
                sample_weight_mode=None, 
                weighted_metrics=None, 
                target_tensors=None):

        self.compile_params = dict(optimizer=optimizer, 
                                   loss=loss, 
                                   metrics=metrics, 
                                   loss_weights=loss_weights, 
                                   sample_weight_mode=sample_weight_mode, 
                                   weighted_metrics=weighted_metrics, 
                                   target_tensors=target_tensors)

        if type(optimizer) == str:
            self.optimizer_config[optimizer] = getattr(keras.optimizers, optimizer).get_config()
        elif issubclass(type(optimizer), keras.optimizers.Optimizer):
            self.optimizer_config[class_name(optimizer)] = optimizer.get_config()
        else:
            raise ValueError('The optimizer must be an instance of a class in keras.optimizers or a string with its name')

        if type(loss) == str:
            self.loss_config[loss] = getattr(keras.optimizers, loss).get_config()
        elif issubclass(type(loss), keras.losses.Loss):
            self.optimizer_config[class_name(loss)] = loss.get_config()
        else:
            raise ValueError('The los must be an instance of a class in keras.losses or a string with its name')
        

    def fit(self, x=None, 
            y=None, 
            batch_size=None, 
            epochs=1, 
            n_inits=1,
            init_metric = 'val_accuracy',
            verbose=1, 
            callbacks=None, 
            validation_split=0.0, 
            validation_data=None, 
            shuffle=True, 
            class_weight=None, 
            sample_weight=None, 
            initial_epoch=0, 
            steps_per_epoch=None, 
            validation_steps=None, 
            validation_freq=1, 
            max_queue_size=10, 
            workers=1, 
            use_multiprocessing=False):
        """Alias for the fit method from keras.models.Sequential with multiple initializations.
        All parameters except the ones below function exactly like the ones from the cited method
        and are applied to every initialization.
           
        Parameters:
        
        n_inits: int
            Number of initializations of the model
        
        init_metric: str
            Name of the metric mesured during the fitting of the model that will be used to select
            the best method

        Returns:
        
        best_log: keras.callbacks.callbacks.History
            History callback from the best model
        
        best_init: int
            Number of the best initialization statring from zero
        """

        if verbose:
            print('Starting the multiple initializations')

        for init in range(n_inits):
            current_model = self._build_model()
            if verbose:
                print('---------------------------------------------------------------------------------------------------')
                print(f'Starting initialization {init}')
            current_log = current_model.fit(x=x, 
                                            y=y, 
                                            batch_size=batch_size, 
                                            epochs=epochs, 
                                            verbose=verbose, 
                                            callbacks=callbacks, 
                                            validation_split=validation_split, 
                                            validation_data=validation_data, 
                                            shuffle=shuffle, 
                                            class_weight=class_weight, 
                                            sample_weight=sample_weight, 
                                            initial_epoch=initial_epoch, 
                                            steps_per_epoch=steps_per_epoch, 
                                            validation_steps=validation_steps, 
                                            validation_freq=validation_freq, 
                                            max_queue_size=max_queue_size, 
                                            workers=workers, 
                                            use_multiprocessing=use_multiprocessing)
            
            if init == 0:
                best_model = current_model
                best_log = current_log
                best_init = 0
            else:
                if best_log.history[init_metric][-1] < current_log.history[init_metric][-1]:
                    best_model = current_model
                    best_log = current_log
                    best_init = init
            
            gc.collect()    #Collecting the discarded model instance

        self.model = best_model
        self.trained = True

        return best_log, best_init   
        
    def evaluate(self, x=None, 
                 y=None, 
                 batch_size=None, 
                 verbose=1, 
                 sample_weight=None, 
                 steps=None, 
                 callbacks=None, 
                 max_queue_size=10, 
                 workers=1, 
                 use_multiprocessing=False):
        """Alias of keras Sequential.evaluate method"""
        return self.model.evaluate(
                x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
                callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        """Alias of keras Sequential.predict method"""
        return self.model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):
        raise NotImplementedError

    def predict_on_batch(self, x):
        raise NotImplementedError

    def fit_generator(self, generator, 
                      steps_per_epoch=None, 
                      epochs=1, 
                      n_inits=1,
                      init_metric='val_accuracy',
                      verbose=1, 
                      callbacks=None, 
                      validation_data=None, 
                      validation_steps=None, 
                      validation_freq=1, 
                      class_weight=None, 
                      max_queue_size=10, 
                      workers=1, 
                      use_multiprocessing=False, 
                      shuffle=True, 
                      initial_epoch=0):
        """Alias for the fit_generator method from keras.models.Sequential with multiple initializations.
        All parameters except the ones below function exactly like the ones from the cited method
        and are applied to every initialization.
           
        Parameters:
        
        n_inits: int
            Number of initializations of the model
        
        init_metric: str
            Name of the metric mesured during the fitting of the model that will be used to select
            the best method

        Returns:
        
        best_log: keras.callbacks.callbacks.History
            History callback from the best model
        
        best_init: int
            Number of the best initialization statring from zero
        """

        if verbose:
            print('Starting the multiple initializations')

        for init in range(n_inits):
            current_model = self._build_model()

            if verbose:
                print('---------------------------------------------------------------------------------------------------')
                print(f'Starting initialization {init}')

            current_log = current_model.fit_generator(generator, 
                                              steps_per_epoch=steps_per_epoch, 
                                              epochs=epochs, 
                                              verbose=verbose, 
                                              callbacks=callbacks, 
                                              validation_data=validation_data, 
                                              validation_steps=validation_steps, 
                                              validation_freq=validation_freq, 
                                              class_weight=class_weight, 
                                              max_queue_size=max_queue_size, 
                                              workers=workers, 
                                              use_multiprocessing=use_multiprocessing, 
                                              shuffle=shuffle, 
                                              initial_epoch=initial_epoch)
            
            if init == 0:
                best_model = current_model
                best_log = current_log
                best_init = 0
            else:
                if best_log.history[init_metric][-1] < current_log.history[init_metric][-1]:
                    best_model = current_model
                    best_log = current_log
                    best_init = init
            
            self.model = best_model
            self.trained = True
            
        return best_log, best_init   

    def get_layer(self, name=None, index=None):
        """Returns a layer based on the parameters.
        If none were passed, returns a list qith all layers
        If both are passed index takes precedence

        Parameters:

        name: str
            Name of the desired layer
        
        index: int
            Index of the desired layer where -1 is the output layer of the model and 0 the input layer
        """

        if (type(index) == type(None)) and (type(name) == type(None)):
            return self.model.layers
        elif type(index) == int:
            return self.model.layers[index]
        elif type(name) == str:
            for layer in self.model.layers:
                try:
                    layer.get_config[name]
                    return layer
                except:
                    pass
        else:
            raise ValueError('The index parameter must be a python int and the name parameter a python string')
    
    def add(self, layer):
        """Alias of keras Sequential.add method"""
        self.layers_config[class_name(layer)] = layer.get_config()

    def save(self,  filepath, overwrite=True, save_format='tf', signatures=None, options=None):
        """Alias of keras Sequential.save method"""

        self.model.save(filepath, overwrite=overwrite, include_optimizer=True, save_format=save_format, signatures=None, options=None)

    @classmethod
    def load(cls, filepath):

        """Loads the model from filepath

        Parameters

        filepath: str
            Location of the model to be loaded
        
        Return

        sequential_model: SequentialModel
            An instance of Sequential Model loaded from the file
        """

        model = load_model(filepath)
        layers_config = OrderedDict()
        optimizer_config = OrderedDict()
        loss_config = OrderedDict()

        for layer in model.layers:
            layers_config[class_name(layer)] = layer.get_config()
        
        optimizer_config[class_name(model.optimizer)] = model.optimizer.get_config()

        loss = getattr(keras.losses, model.loss)()
        loss_config[class_name(loss)] = loss.get_config()

        sequential_model = cls(**layers_config)
        sequential_model.optimizer_config = optimizer_config
        sequential_model.loss_config = loss_config
        sequential_model.trained = True

        return sequential_model

    def _build_model(self):
        model = Sequential()
        for layer_name, layer_parameters in self.layers_config.items():
            layer_parameters = deepcopy(layer_parameters)
            layer_parameters.pop('name')
            layer = getattr(keras.layers, layer_name).from_config(layer_parameters)
            model.add(layer)
        model.compile(**self.compile_params)
        gc.collect()    #Collecting lost layers generated
        return model

