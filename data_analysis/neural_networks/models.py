import gc
import os
import joblib
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from copy import deepcopy
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model, save_model
from data_analysis.utils.utils import frame_from_history

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

		inits_functions: list
			List of functions to be applied to every initialization of the model.
			The functions must accept two arguments, one instance of this class and a filepath
			to the folder output of each init, and return None.

		save_inits: boolean
			If true saves all the models initialized inside a folder called inits_model and allows
			inits_functions to be applied

		Returns:
		
		best_log: keras.callbacks.callbacks.History
			History callback from the best model
		
		best_init: int
			Number of the best initialization statring from zero
		"""

		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)

		#Saving the current model state to reload it multiple times
		blank_dir = os.path.join(cache_dir, 'start_model')
		blank_callbacks = deepcopy(callbacks)
		save_model(model=self._model, filepath=blank_dir)

		if verbose:
			print('Starting the multiple initializations')

		for init in range(n_inits):

			keras.backend.clear_session()   #Clearing old models
			gc.collect()    #Collecting remanescent variables
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

			if verbose:
				print('---------------------------------------------------------------------------------------------------')
				print(f'Starting initialization {init}')

			current_log = current_model.fit(x=x, 
											y=y, 
											batch_size=batch_size, 
											epochs=epochs, 
											verbose=verbose, 
											callbacks=blank_callbacks, 
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
				self._save_log(inits_dir, current_model, current_log)  
				if not inits_functions is None:
					for function in inits_functions:
						function(current_model, inits_dir)

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
		self._save_log(best_dir, best_model, best_log)
		best_init_file = open(os.path.join(best_dir, f'best_init_{best_init}.txt'), 'w')
		best_init_file.close()

		self._model.load_weights(os.path.join(best_dir, 'weights', 'weights'))

		return best_log, best_init 

	@staticmethod
	def _save_log(folderpath, model, history):
		"""Saves multiple parameters of the model state, trainning and topology and itself.

		Parameters:

		folderpath: string
			String with the path to the folder to be saved the files

		model: keras.Sequential
			Keras model with parameters and itself to be saved

		history: keras.callbacks.History
			Callback to have its parameters saved
		"""
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)
		model.save(os.path.join(folderpath, 'model'))
		model.save_weights(os.path.join(folderpath, 'weights', 'weights'))
		params_file = open(os.path.join(folderpath, 'model_params.txt'), 'w')
		params_file.write(f'History.params:\n{history.params}\n')
		for index, layer in zip(range(len(model.layers)), model.layers):
			params_file.write(f'Layer {index} params:\n{layer.get_config()}\n')
		params_file.close()
		joblib.dump(history.params, os.path.join(folderpath, 'callbacks_History_params.joblib'))
		best_frame = frame_from_history(history.history)
		best_frame.to_csv(os.path.join(folderpath, 'training_log.csv'))

	@classmethod
	def load(cls, filepath, custom_objects=None, compile=True):
		model = cls()
		model._model = load_model(filepath, custom_objects, compile)

		return model
		
class ExpertsCommittee():

	def __init__(self, classes, experts=None, wrapper=None):
		self.set_wrapper(wrapper)
		self.experts = dict()
		if experts is None:
			for class_ in classes:
				self.experts[class_] = MultiInitSequential()
		else:
			self._exp_supported(experts)
			for class_, expert in zip(classes, experts):
				self.experts[class_] = expert
		
	def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
			validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
			sample_weight=None, initial_epoch=0, steps_per_epoch=None,
			validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
			use_multiprocessing=False, **kwargs):
		raise NotImplementedError

	def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
		predictions = list()
		for expert in self.experts.values():
			predictions.append(expert.predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, 
											  max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing))
		predictions = np.array(predictions)
		if self.wrapper is None:
			return predictions
		else:
				return self.wrapper.predict(predictions, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks, 
									   max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

	def set_wrapper(self, wrapper):
		if (type(wrapper) == MultiInitSequential):
			self.wrapper = wrapper
		else:
			raise ValueError('Support for classificators other than MultiInitSequential has not been implemented.')

	def add_to_expert(self):
		raise NotImplementedError

	def _check_integrity(self):
		"""It checks if the classificators are correctly built"""
		for class_, expert in self.experts.items():
			if len(expert.layers()) == 0:
				raise ValueError(f'Expert from class {class_} has no layers.')
			elif expert.layers()[-1].get_config()['activation'] != 'tanh':
				raise ValueError(f'Expert from class {class_} must have tanh as activation function on last layer.')

		if not self.wrapper is None:
			if type(self.wrapper) == MultiInitSequential():
				input_shape = self.wrapper.layers()[0].get_config()['batch_input_shape']
				if (len(input_shape)>2) or (input_shape[-1] != len(list(self.experts.values()))):
					raise ValueError(f'The input shape of the wrapper must be the number of experts. Current shape {input_shape[1:]}')
			else:
				raise ValueError('Support for wrapper classificators other than MultiInitSequential has not been implemented.')
			
	@staticmethod
	def _exp_supported(experts):
		"""It checks if the passed experts are supported"""
		try:
			for expert in experts:
				if (type(expert) == MultiInitSequential):
					pass
				else:
					ValueError('Support for experts other than MultiInitSequential has not been implemented.')
		except TypeError:
			if (type(experts) == MultiInitSequential):
				pass
			else:
				ValueError('Support for experts other than MultiInitSequential has not been implemented.')