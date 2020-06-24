import numpy as np
import sklearn as sk

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