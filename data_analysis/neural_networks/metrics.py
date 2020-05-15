"""Modelue that contains metrics compatible with tensorflow keras"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

def sparse_accuracy(y_true, y_pred):
    """Calculates the accuracy of a model with winner takes all classification"""

    typo = y_pred.dtype
    y_true = tf.cast(tf.argmax(y_true, axis=1), dtype=typo)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), dtype=typo)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), dtype=typo))
    total = tf.cast(tf.size(y_true), dtype=typo)

    return tf.divide(correct, total)

def expert_accuracy(y_true, y_pred):
    """Calculates the accuracy of a expert model with tanh as output layer activation function"""

    typo = y_pred.dtype
    y_true = tf.cast(y_true, typo)
    y_pred = tf.cast(y_pred, typo)
    classf = tf.where((1-y_pred) < 1, 1, -1)
    classf = tf.cast(classf, typo)
    correct = tf.reduce_sum(tf.cast(tf.equal(classf, y_true), dtype=typo))
    total = tf.cast(tf.size(y_true), typo)

    return tf.divide(correct, total)