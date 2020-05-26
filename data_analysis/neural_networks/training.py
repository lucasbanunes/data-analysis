import gc
import os
import math
import json
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from data_analysis.utils import utils, metrics

def multi_init_fit(model, compile_params, x, 
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
					mode = 'max',
					inits_functions=None,
					save_inits=False, 
					cache_dir=''):
	"""Alias for the fit method from keras' models with multiple initializations.
	All parameters except the ones below  and the callbacks parameter function 
	exactly like the ones from the cited method and are applied to every initialization.

	With this fit, a keras.callbacks.ModelCheckpoint instance will always be appended to the list of callbacks
	or added if there are none as a way to save and to keep track of the best initialization to be returned

	Parameters:

	model: keras.Model or keras.Sequential
		Model to be trained

	n_inits: int
		Number of initializations of the model

	init_metric: str
		Name of the metric mesured during the fitting of the model that will be used to select
		the best method

	mode: str
		Accepts two values: max and min
		If min is passed, the init_metric will be minimized
		If max is passed, the init_metric will be maxmized

	inits_functions: list
		List of functions to be applied to every initialization of the model.
		The functions must accept two arguments, one instance of this class and a filepath
		to the folder output of each init, and return None.

	save_inits: boolean
		If true saves all the models initialized inside a folder called inits_model and allows
		inits_functions to be applied

	cache_dir: str
		Path to save the models, parameters and temporary data

	Returns:

	model: keras.Model
		Uncompiled trained keras.Model

	log_dict: dict
		Dict with the given values:
		history_callback: keras.callback.History of the best init
		inits_log: dict with informations from the initializations"""

	start_time = time.time()
	inits_time = list()

	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	if verbose:
		print('Starting the multiple initializations')

	for init in range(1, n_inits+1):

		init_start = time.time()

		reinitialize_weights(model, False)

		#Setting the callbacks
		init_dir = os.path.join(cache_dir, 'inits_models', f'init_{init}')
		ck_dir = os.path.join(init_dir, 'models')
		checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(ck_dir, 'epoch_{epoch}'), 
														save_best_only=True, 
														monitor=init_metric, 
														verbose=verbose, mode=mode)
		if callbacks is None:
			init_callbacks = [checkpoint]
		else:
			init_callbacks = deepcopy(callbacks)
			init_callbacks.append(checkpoint)

		model.compile(**deepcopy(compile_params))

		if verbose:
			print('---------------------------------------------------------------------------------------------------')
			print(f'Starting initialization {init}')

		init_callback = model.fit(x=x, 
									y=y, 
									batch_size=batch_size, 
									epochs=epochs, 
									verbose=verbose, 
									callbacks=init_callbacks, 
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

		init_best_epoch = int(np.sort(os.listdir(ck_dir))[-1].split('_')[-1])
		init_best_metric = init_callback.history[init_metric][init_best_epoch-1]
		init_best_model = os.path.join(ck_dir, f'epoch_{init_best_epoch}')

		if save_inits:
			with open(os.path.join(init_dir, 'fitting_params.json'), 'w') as json_file:
				json.dump(utils.cast_dict_to_python(init_callback.params), json_file, indent=4)
			with open(os.path.join(init_dir, 'fitting_metrics.json'), 'w') as json_file:
				json.dump(utils.cast_dict_to_python(init_callback.history), json_file, indent=4)

		#Executing the functions
		if not inits_functions is None:
			for function in inits_functions:
				function(init, model, init_dir)

		#Updating the best model    
		if init == 1 or metrics.optimized(best_metric, init_best_metric, mode):
			best_model = init_best_model
			best_callback = init_callback
			best_init = init
			best_metric = init_best_metric
			best_epoch = init_best_epoch

		init_end = time.time()

		inits_time.append(round((init_end-init_start), 2))

		gc.collect()

	#Defining and saving the best model
	model = load_model(best_model, compile=False)
	best_dir = os.path.join(cache_dir, 'best_model')
	shutil.copytree(best_model, os.path.join(best_dir, 'model'))
	with open(os.path.join(best_dir, 'fitting_params.json'), 'w') as json_file:
		json.dump(utils.cast_dict_to_python(init_callback.params), json_file, indent=4)
	with open(os.path.join(best_dir, 'fitting_metrics.json'), 'w') as json_file:
		json.dump(utils.cast_dict_to_python(init_callback.history), json_file, indent=4)

	if not save_inits:
		shutil.rmtree(os.path.join(cache_dir, 'inits_models'))

	#Saving model params
	with open(os.path.join(cache_dir, 'model_topology.json'), 'w') as json_file:
		json_file.write(model.to_json(indent=4))

	end_time = time.time()
	inits_log = dict(best_init=best_init, best_epoch=best_epoch, elapsed_time=round((end_time-start_time), 2), inits_time=inits_time) 

	with open(os.path.join(cache_dir, f'inits_log.json'), 'w') as json_file:
		json.dump(inits_log, json_file, indent=4)

	gc.collect()
	K.clear_session()

	log = dict(best_callback=best_callback, inits_log=inits_log)

	return model, log

def reinitialize_weights(model, verbose=True):
	"""Reinitialize the trainable weights from the model

	Parameters: 

	model: keras.Model or keras.Sequential
		Model to have its weights reinitialized

	verbose: bool
		If True outputs information from the function execution

	"""
	if verbose:
		print(f'Reinitializing model {model.name} weights')
	for layer in model.layers:
		if layer.trainable:
			layer_config = layer.get_config()
			try:
				#Config dict will have 'layers' as key only if it is a model
				layer_config['layers']
				reinitialize_weights(layer, verbose)
			except KeyError:
				if verbose:
					print(f'Reinitializing layer {layer.name}')
				shapes = [weight.shape for weight in layer.get_weights()]
				param = list()
				for i in range(len(shapes)):
					if i==0:	#Actual weights
						initializer = getattr(keras.initializers, layer_config['kernel_initializer']['class_name']).from_config(layer_config['kernel_initializer']['config'])
					elif i==1:	#Bias
						initializer = getattr(keras.initializers, layer_config['bias_initializer']['class_name']).from_config(layer_config['bias_initializer']['config'])
					param.append(np.array(initializer.__call__(shape=shapes[i])))
				layer.set_weights(param)