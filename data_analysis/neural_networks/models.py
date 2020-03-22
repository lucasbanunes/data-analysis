import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.models import Sequential, clone_model
from data_analysis.functions.utils import class_name


class SequentialModel():
    def __init__(self, **layers):

        #Attributes
        self.model = Sequential()
        self.layers_config = OrderedDict()
        self.compile_config = None

        if layers:
            self.model = Sequential()
            for layer_name, layer_parameters in layers.items():
                layer = getattr(keras.layers, layer_name)(**layer_parameters)
                self.layers_config[layer_name] = layer.get_config()
    
    def compile(self, optimizer, 
                loss=None, 
                metrics=['accuracy'], 
                loss_weights=None, 
                sample_weight_mode=None, 
                weighted_metrics=None, 
                target_tensors=None):#Keras Sequential method

        self.compile_config = dict(optimizer=optimizer, 
                                   loss=loss, 
                                   metrics=metrics, 
                                   loss_weights=loss_weights, 
                                   sample_weight_mode=sample_weight_mode, 
                                   weighted_metrics=weighted_metrics, 
                                   target_tensors=target_tensors)

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
            use_multiprocessing=False):#Keras Sequential method

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

            if verbose:
                print('---------------------------------------------------------------------------------------------------')
            
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
            self.training_log = best_log

        return best_model, best_log, best_init   
        
    def evaluate(self, x=None, 
                 y=None, 
                 batch_size=None, 
                 verbose=1, 
                 sample_weight=None, 
                 steps=None, 
                 callbacks=None, 
                 max_queue_size=10, 
                 workers=1, 
                 use_multiprocessing=False):#Keras Sequential method
        raise NotImplementedError

    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):#Keras Sequential method
        raise NotImplementedError

    def train_on_batch(self, x, y, sample_weight=None, class_weight=None, reset_metrics=True):#Keras Sequential method
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
                      initial_epoch=0):#Keras Sequential method

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
            
            if verbose:
                print('---------------------------------------------------------------------------------------------------')
            
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
            self.training_log = best_log
            
        return best_model, best_log, best_init   

    def evaluate_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):#Keras Sequential method
        raise NotImplementedError

    def predict_generator(self, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):#Keras Sequential method
        raise NotImplementedError

    def get_layer(self, name=None, index=None):#Keras Sequential method
        raise NotImplementedError
    
    def add(self, layer):#Keras Sequential method
        self.layers_config[class_name(layer)] = layer.get_config()

    def get_layers(self):
        return self.model.layers

    def _build_model(self):
        model = Sequential()
        for layer_name, layer_parameters in self.layers_config.items():
            layer_parameters = deepcopy(layer_parameters)
            layer_parameters.pop('name')
            layer = getattr(keras.layers, layer_name)(**layer_parameters)
            model.add(layer)
        model.compile(**self.compile_config)
        return model

