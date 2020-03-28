import gc
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model

class MultiInitSequential(Sequential):
    def __init___(self, layers=None, name=None):
        super(MultiInitSequential, self).__init__(layers=None, name=None)

    def multi_init_fit(self, x=None, 
            y=None, 
            batch_size=None, 
            epochs=1, 
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
            use_multiprocessing=False, 
            n_inits=1,
            init_metric='val_accuracy',
            save_inits=False, 
            cache_dir=''):
        """Alias for the fit method from keras.models.Sequential with multiple initializations.
        All parameters except the ones below function exactly like the ones from the cited method
        and are applied to every initialization.
           
        Parameters:
        
        n_inits: int
            Number of initializations of the model
        
        init_metric: str
            Name of the metric mesured during the fitting of the model that will be used to select
            the best method

        save_inits: boolean
            If true saves all the models initialized inside a folder called inits_model

        Returns:
        
        best_log: keras.callbacks.callbacks.History
            History callback from the best model
        
        best_init: int
            Number of the best initialization statring from zero
        """

        #Saving the current model state to reload it multiple times
        blank_dir = os.path.join(cache_dir, 'blank_model')
        save_model(model=self, filepath=blank_dir)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if verbose:
            print('Starting the multiple initializations')

        for init in range(n_inits):

            keras.backend.clear_session()   #Clearing old models
            gc.collect()    #Collecting remanescent variables
            current_model = load_model(blank_dir)

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
            
            #Saving the initialization model
            if save_inits:
                inits_dir = os.path.join(cache_dir, 'inits_models', f'init_{init}')
                if not os.path.exists(inits_dir):
                    os.makedirs(inits_dir)
                current_model.save(os.path.join(inits_dir, f'init_{init}_model'))
                callback_history = open(os.path.join(inits_dir,f'init_{init}_params.txt'), 'w')
                callback_history.write(f'History.params:\n{current_log.params}\n')
                joblib.dump(current_log.params, os.path.join(inits_dir, 'callbacks_History_params.joblib'))
                log_frame = pd.DataFrame.from_dict(current_log.history)
                log_frame.to_csv(os.path.join(inits_dir, f'init_{init}_training_log.csv'))                

            #Updating the best model    
            if init == 0:
                best_model = current_model
                best_log = current_log
                best_init = 0
            else:
                if best_log.history[init_metric][-1] < current_log.history[init_metric][-1]:
                    best_model = current_model
                    best_log = current_log
                    best_init = init
            
        #Saving the best model
        best_dir = os.path.join(cache_dir, 'best')
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        best_model.save(os.path.join(best_dir, 'best_model'))
        best_model.save_weights(os.path.join(best_dir, 'best_weights', 'best_weights'))
        best_init_file = open(os.path.join(best_dir, f'best_init_{best_init}.txt'), 'w')
        best_init_file.close()
        callback_history = open(os.path.join(best_dir, 'best_init_history.txt'), 'w')
        callback_history.write(f'History.params:\n{best_log.params}\n')
        callback_history.close()
        joblib.dump(best_log.params, os.path.join(best_dir, 'callbacks_History_params.joblib'))
        best_frame = pd.DataFrame.from_dict(best_log.history)
        best_frame.to_csv(os.path.join(best_dir, 'best_training_log.csv'))

        self.load_weights(os.path.join(best_dir, 'best_weights', 'best_weights'))

        return best_log, best_init 