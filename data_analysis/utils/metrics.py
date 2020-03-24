import numpy as np
import tensofrlow as tf
from tensorflow import keras
from tensorflow.keras import backend

def sp_index(y_true, y_pred):
    if backend.is_tensor(y_true) and back.is_tensor(y_pred):
        return TensorMetrics.sp_index(y_true, y_pred)


class TensorMetrics:
    def __init__(self):
        pass
    
    @staticmethod
    def sp_index(y_true, y_pred):
        pass
    
    @staticmethod
    def recall_score(y_true, y_pred):
        pass
     