import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, clone_model

def fit_generator(model, generator, 
                    steps_per_epoch=None, 
                    epochs=1, 
                    n_inits=1,
                    init_metric='accuracy',
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
        print(model.summary())
    for init in range(n_inits):
        current_model = clone_model(model)
        if verbose:
            print('---------------------------------------------------------------------------------------------------')
            print(f'Starting initialization {init+1}')
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
        
        if init == 1:
            best_model = current_model
            best_log = current_log
            best_init = init
        else:
            if best_log.history[init_metric][-1] < current_log.history[init_metric][-1]:
                best_model = current_model
                best_log = current_log
                best_init = init
        
    return best_model, best_log, best_init   