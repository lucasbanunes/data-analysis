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