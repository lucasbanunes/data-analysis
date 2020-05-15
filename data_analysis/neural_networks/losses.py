"""Module that contains losses compatible with tensorflow keras"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

def expert_mse(y_true, y_pred):
    """Calculates mean squared error for a class expert with tanh as output
    layer activation function.
    In this implementation instead of the mse being calculated over all samples, 
    the expert_mse is the mean betweeen the mse for the class and non-class events.
    That insures that the natual unbalance between the number of events is treated
    giving the same weight for each classification.
    """

    typo = y_pred.dtype
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=typo)
    class_mask = tf.equal(y_true, 1)
    non_class_mask = tf.equal(y_true, -1)
    #There may be cases that an entire batch may have events of only one class, the parameter bellow tests if this is the case
    only_one_class = tf.cast(tf.reduce_all(class_mask), dtype=tf.int16) + tf.cast(tf.reduce_all(non_class_mask), dtype=tf.int16)
    #If the batch only has events of only one class then the normal mse is the output
    error = K.switch(only_one_class,
                        keras.losses.mean_squared_error(y_true, y_pred), 
                        0.5*keras.losses.mean_squared_error(y_true[class_mask], y_pred[class_mask])+0.5*keras.losses.mean_squared_error(y_true[non_class_mask], y_pred[non_class_mask]))
    return error