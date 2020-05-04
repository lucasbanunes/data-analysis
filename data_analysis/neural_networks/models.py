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
import tensorflow as tf
from copy import deepcopy
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model
from data_analysis.utils import utils
from data_analysis.neural_networks import training
from data_analysis.utils.utils import frame_from_history, DataSequence, gradient_weights, _WrapperSequence, cast_dict_to_python

class MultiInitSequential():
	"""Alias for keras.Sequential model but with adittional functionalities"""

	def __init__(self, layers=None, name=None):
		self._model = Sequential(layers, name)

	def add(self, layer):
		return self._model.add(layer)

	def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None,
				sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
				distribute=None, **kwargs):

		return self._model.compile(optimizer, loss, metrics, loss_weights,
						   sample_weight_mode, weighted_metrics, target_tensors,
						   distribute, **kwargs)
	
	def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
				 callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
		
		return self._model.evaluate(x, y, batch_size, verbose, sample_weight, steps,
								   callbacks, max_queue_size, workers, use_multiprocessing)

	def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
			validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
			sample_weight=None, initial_epoch=0, steps_per_epoch=None,
			validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
			use_multiprocessing=False, **kwargs):

		return self._model.fit(x, y, batch_size, epochs, verbose, callbacks,
							  validation_split, validation_data, shuffle, class_weight,
							  sample_weight, initial_epoch, steps_per_epoch,
							  validation_steps, validation_freq, max_queue_size, workers,
							  use_multiprocessing, **kwargs)

	def get_layer(self, name=None, index=None):
		return self._model.get_layer(name, index)

	def load_weights(self, filepath, by_name=False, skip_mismatch=False):
		return self._model.load_weights(filepath, by_name, skip_mismatch)

	def pop(self):
		return self._model.pop()

	def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
		return self._model.predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)
	
	def predict_classes(self, x, batch_size=32, verbose=0):
		return self._model.predict_classes(x, batch_size, verbose)

	def predict_on_batch(self, x):
		return self._model.predict_on_batch(x)

	def predict_proba(self, x, batch_size=32, verbose=0):
		return self._model.predict_proba(x, batch_size, verbose)

	def reset_metrics(self):
		return self._model.reset_metrics()

	def reset_states(self):
		return self._model.reset_states()

	def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None):
		return self._model.save(filepath, overwrite, include_optimizer, save_format, signatures, options)
	
	def save_weights(self, filepath, overwrite=True, save_format=None):
		return self._model.save_weights(filepath, overwrite, save_format)

	def summary(self, line_length=None, positions=None, print_fn=None):
		return self._model.summary(line_length, positions, print_fn)

	def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
		return self._model.test_on_batch(x, y, sample_weight, reset_metrics)

	def to_json(self, **kwargs):
		return self._model.to_json(**kwargs)

	def to_yaml(self, **kwargs):
		return self._model.to_yaml(**kwargs)
	
	def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
		return self._model.train_on_batch(x, y, sample_weight, class_weight, reset_metrics)

	def layers(self):
		"Method for accessing the layers of the model"
		return self._model.layers

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
			mode = 'max',
			inits_functions=None,
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
		
		log_dict: dict
			Dict with the given values:
			history_callback: keras.callback.History of the best init
			inits_log: dict with informations from the initializations
		"""

		start_time = time.time()
		inits_time = list()

		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)

		#Saving the current model state to reload it multiple times
		blank_dir = os.path.join(cache_dir, 'start_model')
		save_model(model=self._model, filepath=blank_dir)

		#Removing the current model to avoid conflicts with the multiple initializations
		del self._model
		gc.collect()
		keras.backend.clear_session()

		if verbose:
			print('Starting the multiple initializations')

		for init in range(1, n_inits+1):

			init_start = time.time()

			current_model = load_model(blank_dir)

			#Initialiazing new weights
			for layer in current_model.layers:
				layer_config = layer.get_config()
				shapes = [weight.shape for weight in layer.get_weights()]
				param = list()
				for i in range(len(shapes)):
					if i==0:
						initializer = getattr(keras.initializers, layer_config['kernel_initializer']['class_name']).from_config(layer_config['kernel_initializer']['config'])
					elif i==1:
						initializer = getattr(keras.initializers, layer_config['bias_initializer']['class_name']).from_config(layer_config['bias_initializer']['config'])
					param.append(np.array(initializer.__call__(shape=shapes[i])))
				layer.set_weights(param)

			#Setting the callbacks
			inits_dir = os.path.join(cache_dir, 'inits_models', f'init_{init}')
			ck_dir = os.path.join(inits_dir, 'models')
			checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(ck_dir, 'epoch_{epoch}'), 
														 save_best_only=True, 
														 monitor=init_metric, 
														 verbose=1, mode=mode)
			if callbacks is None:
				init_callbacks = [checkpoint]
			else:
				init_callbacks = deepcopy(callbacks)
				init_callbacks.append(checkpoint)

			if verbose:
				print('---------------------------------------------------------------------------------------------------')
				print(f'Starting initialization {init}')

			init_callback = current_model.fit(x=x, 
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

			init_best_epoch = int(os.listdir(ck_dir)[-1].split('_')[-1])
			init_best_metric = init_callback.history[init_metric][init_best_epoch-1]
			init_best_model = os.path.join(ck_dir, os.listdir(ck_dir)[-1])

			if save_inits:
				self._save_history(inits_dir, init_callback)
			
			#Executing the functions
			if not inits_functions is None:
				for function in inits_functions:
					function(current_model, inits_dir)

			#Updating the best model    
			if init == 1 or self._optimized(best_metric, init_best_metric, mode):
				best_model = init_best_model
				best_callback = init_callback
				best_init = init
				best_metric = init_best_metric
				best_epoch = init_best_epoch

			del current_model
			gc.collect()    #Collecting remanescent variables
			keras.backend.clear_session()   #Restarting the graph

			init_end = time.time()

			inits_time.append(round((init_end-init_start), 2))

		#Defining the best model
		self._model = load_model(best_model)
		best_dir = os.path.join(cache_dir, 'best_model')
		self._model.save(os.path.join(best_dir, 'model'))

		if not save_inits:
			shutil.rmtree(os.path.join(cache_dir, 'inits_models'))

		#Saving moel params
		with open(os.path.join(cache_dir, 'model_topology.json'), 'w') as json_file:
			json_file.write(self._model.to_json(indent=4))

		self._save_history(best_dir, best_callback)

		end_time = time.time()
		inits_log = dict(best_init=best_init, best_epoch=best_epoch, elapsed_time=round((end_time-start_time), 2), inits_time=inits_time) 

		with open(os.path.join(cache_dir, f'inits_log.json'), 'w') as json_file:
			json.dump(inits_log, json_file, indent=4)

		return dict(best_callback=best_callback, inits_log=inits_log) 

	@staticmethod
	def _save_history(folderpath, callback):
		"""Saves the dicts from keras.calbacks.History as json files"""
		with open(os.path.join(folderpath, 'fitting_params.json'), 'w') as json_file:
			json.dump(cast_dict_to_python(callback.params), json_file, indent=4)
		with open(os.path.join(folderpath, 'fitting_metrics.json'), 'w') as json_file:
			json.dump(cast_dict_to_python(callback.history), json_file, indent=4)

	@staticmethod
	def _optimized(best, current, mode):
		if mode == 'max':
			if best < current:
				return True
			else:
				return False
		elif mode == 'min':
			if best > current:
				return True
			else:
				return False
		else:
			raise ValueError(f'Mode {mode} is not supported')

	@classmethod
	def load(cls, filepath, custom_objects=None, compile=True):
		"""Returns a keras.Sequential model as a MultiInitSequential instance.
		All the paramaters work as the same in keras.models.load_model.""" 
		model = cls()
		model._model = load_model(filepath, custom_objects, compile)

		return model
		
class ExpertsCommittee():
	"""Class that implements a Committe Machine as described in
	HAYKIN, S. Neural Networks - A Comprehensive Foundation. Second edition. Pearson Prentice Hall: 1999
	with expert models.

	Attributes:

	experts: dict
		Dict with the classes as keys and each one with the path to the folder
		where the respective expert model is saved.

	wrapper: str
		Path to the folder where the classificator that will take the output of all experts
		and process them to a final output is saved.

	self.mapping: function
		Function that maps the classes to its respective labels.
	"""
	def __init__(self, mapping, classes, experts, wrapper=None, cache_dir=''):
		"""
		Parameters:

		mapping: function
			Function that maps each class to its respective label to be evaluated by the committee.

		classes: iterable
			Iterable with the classes as each item
		
		experts: iterable
			Iterable with each item being a uncompiled keras.Model, or a path to a folder where a
			keras.Model was saved with keras.Model.save or keras.models.save_model.

		wrapper: str or uncompiled keras.Model
			Classificator to be used to process the output of the experts.
			The string must be a path to the folder where a keras.Model was saved.
		"""

		self.cache_dir = cache_dir + 'expert_committee'
		if callable(mapping):
			self.mapping = mapping
		else:
			raise ValueError('The mapping parameter must be callable.')
		self.experts = self._exp_formatting(classes, experts)
		self.wrapper = self._wrap_formatting(wrapper)
		self.compile_params = None

	def compile(self, compile_params):
		"""Method that saves the committee compile parameters to be used when fit is called

		Parameters:

		compile_params: dict
		Dictionary that maps the compile params from keras.Model.compile to each model in the committe.
		The dict must have a committe key that has the compile parameters mapping, and a key for each
		class that has an expert with a mapping for the compila parameters to be used with the expert of that class.
		"""
		self.compile_params = compile_params
		
		
	def fit(self, x=None, epochs=1, verbose=1, callbacks=None,
			validation_data=None, shuffle=True, class_weight=None,
			sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, 
			validation_freq=1, max_queue_size=10, workers=1,use_multiprocessing=False,
			n_inits=1, init_metric='val_accuracy', inits_functions=None, save_inits=False, 
			cache_dir='', **kwargs):

			"""Trains the committee with the given parameters following the same rules
			of MultiInitSequential.multi_init_fit with exception of the class_weight and sample_weight parameters handling.

			Parameters:

			class_weight: dict
				Dict with the keys being 'expert' and 'wrapper' with the desired class_weight for each key.
				Those parameters will be applied to the experts and the wrapper respectively.
				The string 'gradient_weights' can be passed as a value. This will make the class_weight be 
				evaluated by data_analysis.utils.utils.gradient_weights.
				Defaults to None.

			sample_weight: dict
				Dict with the keys being 'expert' and 'wrapper' with the desired sample_weight for each key.
				Those parameters will be applied to the experts and the wrapper respectively.
				Defaults to None.
			
			Returns:

			log: dict
				Returns a log with the logs of each model returned from multi_init_fit
			"""

			self._is_compiled()
			experts_logs = dict()

			for class_, expert_path in self.experts.items():

				if verbose:
					print(f'Training expert for class {class_}')
					
				gc.collect()

				expert = MultiInitSequential()
				expert = load_model(expert_path)
				expert.compile(**self.compile_params[class_])

				cache_dir = os.path.join(self.cache_dir, 'experts', f'{class_}_expert')
				
				train_expert = x[class_]
				val_expert = validation_data[class_]

				if class_weight is None:
					cls_weight = None
				elif (class_weight == 'gradient_weights') or (class_weight['expert'] == 'gradient_weights'):
					cls_weight = train_expert.gradient_weights()
				else:
					cls_weight = class_weight['expert']

				if sample_weight is None:
					spl_weight = None
				else:
					spl_weight = sample_weight['expert']

				experts_logs[class_] = training.multi_init_fit(expert, x=train_expert, epochs=epochs, verbose=verbose, callbacks=callbacks,
										validation_data=val_expert, shuffle=shuffle, class_weight=cls_weight,
										sample_weight=spl_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, 
										validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,use_multiprocessing=use_multiprocessing,
										n_inits=n_inits, init_metric=init_metric, inits_functions=inits_functions, save_inits=save_inits, 
										cache_dir=cache_dir, **kwargs)

				self.experts[class_] = os.path.join(cache_dir, 'best_model', 'model')

			if self.wrapper is None:
				log = dict(wrapper_log=None, experts_logs=experts_logs)
				return logs
			else:

				if verbose:
					print('Training wrapper')

				#Loading the models and connecting them to mount the committee
				committee_input = keras.Input(shape=x.input_shape(), name='committee_input')
				experts = [load_model(expert_path, compile=False)(committee_input) for expert_path in self.experts.values()]
				concat = keras.layers.concatenate(experts)
				wrapper = load_model(self.wrapper, compile=False)(concat)
				committee = keras.Model(committee_input, wrapper, name='expert_committee')

				#Freezing the experts
				for layer in committee.layers[:-1]:		#Leaves the wrapper outside since it is the one to be trained
					if type(layer) == keras.Model:
						layer.trainable=False

				committee.compile(**self.compile_params['committee'])

				cache_dir = os.path.join(cache_dir, 'committee')
				log = dict(experts=experts_logs)

				if class_weight is None:
					cls_weight = None
				elif (class_weight == 'gradient_weights') or (class_weight['committee'] == 'gradient_weights'):
					cls_weight = x.gradient_weights()						
				else:
					cls_weight = class_weight['committee']

				if sample_weight is None:
					spl_weight = None
				else:
					spl_weight = sample_weight['committee']

				log['committee'] = training.multi_init_fit(committee, x=x, y=None, batch_size=None, epochs=epochs, verbose=verbose, callbacks=callbacks,
										validation_split=0.0, validation_data=validation_data, shuffle=shuffle, class_weight=cls_weight,
										sample_weight=spl_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, 
										validation_freq=validation_freq, max_queue_size=max_queue_size, workers=workers,use_multiprocessing=use_multiprocessing,
										n_inits=n_inits, init_metric=init_metric, inits_functions=inits_functions, save_inits=save_inits, 
										cache_dir=cache_dir, **kwargs)

				return committee, log

	def _is_compiled(self):
		"""Checks if compile_params exists if not raises an error"""
		if self.compile_params is None:
			raise RuntimeError('The model must be compiled before training.')
		return wrapper_train, wrapper_val

	def _exp_formatting(self, classes, experts):
		"""It checks if the passed experts are correctly typed and returns
		a correctly formated dict"""
		new_experts = dict()
		classes, experts = utils.sort_pair(classes, experts)
		for class_, expert in zip(classes, experts):
			if type(expert) == keras.Model:
				save_dir = os.path.join(self.cache_dir, 'blank_models', f'{class_}_expert')
				expert.save(save_dir)
				new_experts[class_] = save_dir
				#This removes the model from the graph
				del expert
				gc.collect()
				keras.backend.clear_session()
			elif type(expert) == str:
				new_experts[class_] == expert
			else:
				raise ValueError(f'The expert must be a path to the model folder or the model itself')
		return new_experts