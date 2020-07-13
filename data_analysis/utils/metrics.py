import os
import numpy as np
import pandas as pd
import sklearn as sk
from itertools import product
import matplotlib.pyplot as plt
from data_analysis.utils import utils

def sp_index(y_true, y_pred, classes=None):
    """Calculates the Sum-product index for the given predicted data.

    Parameters:

    y_true: numpy.ndarray
        True classification of the data

    y_pred: numpy.ndarray
        Predicted classification of the data
    
    classes: numpy.ndarray
        Classes to be considered, defaults to the unique values in y_true

    Returns:

    sp: numpy.float64
        The computed metric
    """
    
    if classes is None:
        classes = np.unique(y_true)

    num_classes = len(classes)
    recall_score = sk.metrics.recall_score(y_true, y_pred, labels=classes)
    sp = np.sqrt((np.sum(recall_score)/num_classes)*(np.power(np.product(recall_score), 1/num_classes)))

    return sp

def optimized(best_metric, current_metric, mode):
    """Checks if a metric has been optimized based on the given mode

    Parameters:

    best_metric: int or float
        Best value so far

    current_metric: int or float
        Value to contest with best to see if it is better than the current best one

    mode: str
        If max the value is to be maximized, therefore, returns True if the current is higher that the best.
        If min the value is to be minimized, therefore, returns True if the current is lower than the best.
    
    Raises:

    ValueError:
        If a string not contained in the mode parameters is passed.

    Returns:

    result: bool
        The result of the comparison.
    """
    
    if mode == 'max':
        if best_metric < current_metric:
            return True
        else:
            return False
    elif mode == 'min':
        if best_metric > current_metric:
            return True
        else:
            return False
    else:
        raise ValueError(f'Mode {mode} is not supported')

def confusion_matrix_frame(true, pred, normalize=None, filepath=None):
    classes = np.unique(true)
    cm = sk.metrics.confusion_matrix(true, pred, normalize=normalize)
    cm_frame = pd.DataFrame(cm, index=classes, columns=classes)
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        cm_frame.to_csv(filepath)
    return cm_frame

def plot_confusion_matrix(cm, classes_names=None, cmerr=None, filepath=None):
    if type(cm) is pd.DataFrame:
        classes_names = np.array(cm.columns.values, dtype=str)
        cm = cm.values
    
    if type(cmerr) is pd.DataFrame:
        cmerr = cmerr.values
    n_classes = len(classes_names)
    
    fig, axis = plt.subplots()
    im = axis.imshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(im)
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    thresh = (cm.max() + cm.min())/2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i,j] < thresh else cmap_min
        if cmerr is None:
            text = str(np.around(cm[i,j], 2))
        else:
            err, decimals = utils.around(cmerr[i,j])
            value = np.around(cm[i,j], decimals)
            text = fr"{value} $\pm$ {err}"
        axis.text(j, i, text, ha='center', va='center', color=color)
    
    axis.set(xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=classes_names,
            yticklabels=classes_names,
            ylabel="True label",
            xlabel="Predicted label",
            title='Confusion Matrix')
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(filepath, dpi=200, format='png')
    return fig, axis