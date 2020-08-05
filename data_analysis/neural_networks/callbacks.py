from tensorflow.keras.callbacks import Callback
import numpy as np

class MultiInit(Callback):

    def __init__(self):
        super(MultiInit, self).__init__()
        self.inits = list()
        self.current_init = 0

    def on_train_begin(self, logs=None):
        if self.current_init == 0:
            self.json_config = self.model.to_json(indent=4)
        else:
            if self.json_config != self.model.to_json(indent=4):
                raise ValueError(f'The multi init must be used with only one model configuration. The current model is different from the previous ones')

    def on_train_end(self, logs=None):
        self.inits.append({'weights': self.model.get_weights(), 'logs': None, 'params': None})
        self.current_init += 1

    def add_history(self, history, init=None):

        if init is None:
            init=-1
        
        self.inits[init]['logs'] = history.history
        self.inits[init]['params'] = history.params

    def get_best_init(self, metric, mode, training_end):

        self._check_inits_integrity()
        best_init = 0

        if mode == 'max':
            best_metric = -np.inf
            is_best = lambda x: x>best_metric
            get_best = np.amax
        elif mode == 'min':
            best_metric = np.inf
            is_best = lambda x: x<best_metric
            get_best = np.amin
        else:
            raise ValueError(f'Only "max" and "min" are options for the mode parameter. {mode} was passed')

        if training_end:
            for init in range(len(self.inits)):
                current_metric = self.inits[init]['logs'][metric][-1]
                best_init = init if is_best(current_metric) else best_init
        else:
            for init in range(len(self.inits)):
                current_metric = get_best(self.inits[init]['logs'][metric])
                best_init = init if is_best(current_metric) else best_init
        
        return best_init

    def best_weights(self, metric, mode, training_end):
        best_init = self.get_best_init(metric, mode, training_end)
        return self.inits[best_init]['weights']  


    def _check_inits_integrity(self):
        for init in range(len(self.inits)):
            if self.inits[init]['logs'] is None or self.inits[init]['params'] is None:
                raise ValueError(f'Init {init} is missing its callback.History. You can add it using add_history method')

