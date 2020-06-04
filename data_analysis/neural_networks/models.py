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

		self.compile_params = dict(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights,
									sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors,
									distribute=distribute, **kwargs)

		return self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights,
						   sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors,
						   distribute=distribute, **kwargs)
	
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

		self._model, log = training.multi_init_fit(self._model, self.compile_params, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data,
													shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
													validation_freq, max_queue_size, workers, use_multiprocessing, n_inits, init_metric, 
													mode, inits_functions, save_inits, cache_dir)
		
		self._model.compile(**self.compile_params)

		return log

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

	cache_dir: str
		Path to where to save the builder's cache
	"""
	def __init__(self, classes, experts, cache_dir=''):
		"""
		Parameters:

		classes: iterable
			Iterable with the classes as each item
		
		experts: iterable
			Iterable with each item being a uncompiled keras.Model, or a path to a folder where a
			keras.Model was saved with keras.Model.save or keras.models.save_model.

		cache_dir: str
			Path to where to save the builder's cache
		"""

		self.cache_dir = os.path.join(cache_dir, 'expert_committee')
		self.experts = self._exp_formatting(classes, experts)
		self.compile_params = None

	def compile(self, compile_params):
		"""Method that saves the committee compile parameters to be used when fit is called

		Parameters:

		compile_params: dict
		The dict must have each class as a key and its respective value the compile mapping for the expert
		of that class to be used when fit is called
		"""
		self.compile_params = compile_params
		
		
	def fit(self, x=None, epochs=1, verbose=1, callbacks=None, validation_data=None, class_weight=None,
			validation_freq=1, n_inits=1, init_metric='val_accuracy', mode='max',
			inits_functions=None, save_inits=False, **kwargs):

			"""Trains the experts independently then ensembles them in one keras.Model instance named
			committee
			All the models are trained with multiple initializations
			With the exception of the callbacks parameter the others follow the rules bellow:
			The parameters not listed bellow are applied to all trainings equally and work exactly as in
			keras.Model.fit.
			The ones listed bellow must all be dicts with the keys being the name of the classes of each expert
			the parameter applied and described bellow is the value of that key.
			Example:
			x must be a dict with keys being the name of the classes and the value of each class must be
			a instante or child class of data_analysis.utils.DataSequence with the training data

			Parameters:

			x: Instante or child class of data_analysis.utils.DataSequence
				Training data

			callbacks: list of keras' callbacks
				Callbacks to be applied to the training.
				Since the models are trained with multiple intializations, a keras.callbacks.ModelCheckpoint
				is always applied even if None is passed.
			
			validation_data: dict
				Validation data

			class_weight: dict
				Class weights to be applied to the loss gradient.

			init_metric: str
				Name of the metric to be optimized within the initialization

			mode: dict
				Accepts two values;
				If max the init_metrics will be maximied
				If min the init_metric will be minimized

			"""

			self._is_compiled()
			experts_logs = dict()

			for class_, expert_path in self.experts.items():

				if verbose:
					print(f'Training expert for class {class_}')
					
				gc.collect()

				data_shape = x[class_].input_shape()

				expert = load_model(expert_path)
				expert.compile(**self.compile_params[class_])
				cache_dir = os.path.join(self.cache_dir, 'experts', f'{class_}_expert')
				
				expert, experts_logs[class_] = training.multi_init_fit(compile_params=self.compile_params[class_], model=expert, x=x[class_], epochs=epochs, verbose=verbose, callbacks=callbacks,
														validation_data=validation_data[class_], class_weight=class_weight[class_],
														validation_freq=validation_freq, n_inits=n_inits, init_metric=init_metric[class_], 
														inits_functions=inits_functions, mode=mode[class_],
														save_inits=save_inits, cache_dir=cache_dir, **kwargs)

				self.experts[class_] = os.path.join(cache_dir, 'best_model', 'model')

				del expert
				gc.collect()
				keras.backend.clear_session()

			committee_input = keras.Input(shape=data_shape, name='committee_input')
			experts_list = list()
			for class_, expert_path in self.experts.items():
				expert = load_model(expert_path, compile=False)
				expert.compile(**self.compile_params[class_])
				experts_list.append(expert(committee_input))
			concat = keras.layers.concatenate(experts_list)
			committee = keras.Model(committee_input, concat, name='experts_committee')
			return committee, experts_logs

	def _is_compiled(self):
		"""Checks if compile_params exists if not raises an error"""
		if self.compile_params is None:
			raise RuntimeError('The model must be compiled before training.')

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