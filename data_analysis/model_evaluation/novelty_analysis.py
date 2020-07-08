import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score
from data_analysis.utils import math_utils, utils

def svm_get_results(predictions, labels, nov_label, classes_names, filepath=None):
    """Generates a frame with the results for novelty detection and classification for a 
    SVM committee using One vs Rest classification.

    Parameters:
    
    predictions: numpy.ndarray
        Array with shape (n_samples, n_classes) with the predictions from the committee
    
    labels: numpy.ndarray
        True labels for predictions
    
    nov_label: int
        Label of the novelty data

    classes_names: 1-d array like
        Array with the classes names according to predictions. Column 0 is the predictions
        for the class classes_names[0]
    
    filepath: str
        Filepath to save a csv file of the frame

    Returns:
    
    results_frame: pandas.DataFrame
        Frame with the data having the predictions array, and more two columns for true and 
        predicted labels
    """
    
    supported_types = [np.int16, np.int32, np.int64]
    if not labels.dtype in supported_types:
        raise TypeError(f'The labels array must be of the following types {supported_types} the type {labels.dtype} was passed')
    del supported_types

    novelty_detec = np.all(predictions == -1, axis=1)
    classf = classes_names[np.argmax(predictions, axis=1)]
    nov_and_classf = np.where(novelty_detec, 'Nov', classf).reshape(-1, 1)
    columns = [f'Class_{class_}' for class_ in classes_names]
    columns.extend(['True', 'Pred'])
    classes_names = list(classes_names)
    classes_names.insert(nov_label, 'Nov')
    classes_names = np.array(classes_names)
    y_true = classes_names[labels].reshape(-1, 1)
    data = np.concatenate((predictions, y_true, nov_and_classf), axis=1)
    results_frame = pd.DataFrame(data, columns=columns)

    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        results_frame.to_csv(filepath, index = False)

    return results_frame

def svm_evaluate_nov_detection(results_frame, metrics=None, filepath=None):
    """Evaluates accuracy for each class and other custom metrics as dict where each key is the metric name
    and its value the computed metric.

    Parameters:

    results_frame: pandas.DataFrame
        The frame returned from dvm_get_results

    metrics: list
        List where each item is a cllable thar represents a function that recieves y_true and y_pred as arguments

    filepath: str
        If passed, it becomes the path to save the json file of the dict

    Returns:

    evaluation_dict: dict
        Dict with the metrics evaluated
    """

    y_true = results_frame.loc[:, 'True'].values.flatten()
    y_pred = results_frame.loc[:, 'Pred'].values.flatten()
    evaluation_dict = {f'{class_}_acc': accuracy_score(np.where(y_true == class_, 1, 0), np.where(y_pred == class_, 1, 0))
                        for class_ in np.unique(y_true)}
    if not metrics is None:
        for metric in metrics:
            evaluation_dict[metric.__name__] = metric(y_true, y_pred)
    
    if not filepath is None:
        with open(filepath, 'w') as json_file:
            json.dump(utils.cast_to_python(evaluation_dict), json_file, indent=4)

    return evaluation_dict

def create_threshold(quantity, 
                     interval):
        """
        Creates a threshold array for the model using np.linspace
        
        Parameters

        quantity: iterable
            Each position has the number of values for each interval in the same position
        interval: iterable
            Each position has a len 2 iterable with the interval where to extract the corresponding quantity

        Return

            threshold: numpy.ndarray
        """
        threshold = np.array(list())
        for q, i in zip(quantity, interval):
            threshold = np.concatenate((threshold, np.linspace(i[0], i[1], q)))
        return threshold

def get_results(predictions,
                labels,
                threshold,
                classes_names,
                novelty_index,
                filepath = None):
        """
        Creates a pandas.DataFrame with the events as rows and the output of the neurons in the output layer, 
        the classification for each threshold and the labels as columns.
        It uses winner takes all method for classification and a threshold on the output layer for 
        novelty detection.

        Parameters

        predictions: numpy array
            Output from the output layer

        labels: numpy array
            Correct data label.
            The arraay must be filled with ints which each value i represents the class with name classes_names[i]

        threshold: numpy array
            Array with the threshold values

        classes_names: 1-d array like
            Array with the name of the classes

        novelty_index: int
            Number of the label that is treated as novelty

        filepath: string
            Where to save the .csv file. Defaults to None and if that happens the frame is not saved.

        Returns:

        novelty_frame: pandas.DataFrame
            Frame with all the data with the following organization:
            Events as rows
            Two layers of columns:
                Neurons: predictions data where each subcolumn is named out0, out1, etc
                Classification: the model classification with the threshold as subcolumns
                Labels: with one subcolumn L with the correct labels of the data set
        """
        classes_names = list(np.array(classes_names))
        classes_names.pop(novelty_index)
        novelty_matrix = _get_novelty_matrix(predictions, threshold, np.array(classes_names))
        classes_names.insert(novelty_index, 'Nov')
        classes_names = np.array(classes_names)
        labels_matrix = classes_names[labels].reshape(-1,1)
        outer_level = list()
        inner_level = list()
        for i in range(predictions.shape[-1]):
            outer_level.append('Neurons')
            inner_level.append(f'out{i}')
        for t in threshold:
            outer_level.append('Classification')
            inner_level.append(t)
        outer_level.append('Labels')
        inner_level.append('L')
        novelty_frame = pd.DataFrame(np.concatenate((predictions, novelty_matrix, labels_matrix), axis = 1), columns = pd.MultiIndex.from_arrays([outer_level, inner_level]))
        if not filepath is None:
            folder, _ = os.path.split(filepath)
            if not os.path.exists(folder):
                os.makedirs(folder)
            novelty_frame.to_csv(filepath, index = False)
        return novelty_frame

def evaluate_nov_detection(results_frame,
                    metrics = None, 
                    filepath = None):
    """
    Creates a data frame with accuracy evaluated for novelty detection, classification and for classfication
    of each class, trigger rate and novelty rate. Other metrics can be passed and added to the frame.

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results function
    
    metrics: list
        functions that recieve results_frame as a parameter and return a array with it metric calculated
        for each threshold.

    filepath: string
        Where to save the frame as a .csv file. Defaults to None and if that happens the frame is not saved.
    
    Return
        eval_frame: pandas.DataFrame
    """
    columns = results_frame.loc[:, 'Classification'].columns.values.flatten()
    classes = np.unique(results_frame.loc[:, 'Labels'].values.flatten())
    known_classes = classes[classes != 'Nov']
    index = ['Nov acc', 'Classf acc', 'Trigger rate', 'Nov rate']
    data = [nov_accuracy_score(results_frame), classf_accuracy_score(results_frame)]
    data.extend(list(get_recall_score(results_frame)))
    for class_ in known_classes:
        index.append(f'{class_} Acc')
        data.append(_class_acc_per_class(results_frame, class_))
    if not metrics is None:
        for metric in metrics:
            index.append(metric.__name__)
            data.append(metric(results_frame))
    eval_frame = pd.DataFrame(data, index=index, columns=columns)
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        eval_frame.to_csv(filepath, index = False)
    return eval_frame

def plot_noc_curve(results_frame, nov_class_name, figsize=(12,3), area=True, filepath=None):
    """Plot noc curve fro given results_frame

    Parameters:

    results_frame: pandas.DataFrame
        DataFrame returned from get_results function

    nov_class_name: str
        Name of the novelty class

    figsize: tuple
        Tuple with the size of the matplotlib.Figure

    area: bool
        If true calculates  and marks the auc showing its value

    filepath: str
        Path to save the plot. Dafaults to None and if so the figure is not saved.

    Returns:
        
    fig: matplotlib.Figure
        Figure with the plot
    
    axis:
        axis with the plot
    """

    y_true = np.where(results_frame.loc[:, 'Labels'].values.flatten() == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:,'Classification'] == 'Nov', 1, 0)
    trigger, novelty_rate = np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None), 0, novelty_matrix)
    fig, axis = plt.subplots(figsize=figsize)
    axis.set_title(f'NOC Curve Novelty class {nov_class_name}')
    axis.set_ylabel('Trigger Rate')
    axis.set_ylim(0,1)
    axis.set_xlim(0,1)
    axis.set_xlabel('Novelty Rate')
    axis.plot(novelty_rate, trigger, color='k')
    axis.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    if area:
        noc_area = math_utils.trapezoid_integration(novelty_rate, trigger)
        noc_area = round(noc_area, 3)
        axis.fill_between(novelty_rate, trigger, interpolate=True, color='#808080')
        plt.text(0.3, 0.25, f'Area = {noc_area}',
                    horizontalalignment='center', fontsize=20)
    plt.tight_layout()
    if not filepath is None:
        fig.savefig(fname=filepath, dpi=200, format='png')
    plt.close(fig)
    return fig, axis

def plot_accuracy_curve(results_frame, nov_class_name, figsize=(12,3), filepath=None):
    """Plot nov acc and classf acc curves for given results_frame

    Parameters:

    results_frame: pandas.DataFrame
        DataFrame returned from get_results function

    nov_class_name: str
        Name of the novelty class

    figsize: tuple
        Tuple with the size of the matplotlib.Figure

    filepath: str
        Path to save the plot. Dafaults to None and if so the figure is not saved.

    Returns:
        
    fig: matplotlib.Figure
        Figure with the plot
    
    axis:
        axis with the plot
    """
    nov_acc = nov_accuracy_score(results_frame)
    classf_acc = classf_accuracy_score(results_frame)
    threshold = results_frame.loc[:,'Classification'].columns.values.flatten()
    fig, axis = plt.subplots(figsize=figsize)
    axis.set_title(f'Novelty class {nov_class_name}')
    axis.set_ylabel('Accuracy')
    axis.set_ylim(0,1)
    axis.set_xlabel('Threshold')
    axis.plot(threshold, nov_acc, color='blue', label='Nov acc')
    axis.plot(threshold, classf_acc, color='red', label='Classf acc')
    axis.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
    axis.legend(loc=4)
    plt.tight_layout()
    if not filepath is None:
        fig.savefig(fname=filepath, dpi=200, format='png')
    plt.close(fig)
    return fig, axis

def get_recall_score(results_frame):
    """Computes recall_score for each threshold for both novelty detection and known data.
        Returns a (len(threshold), 2) shape ndarray where index[0] is recall for known class and index[1] for novelty class."""
    y_true, novelty_matrix = _get_as_binary(results_frame)
    return np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None), 0, novelty_matrix)

def nov_accuracy_score(results_frame):
    """Computes accuracy_score for each threshold"""
    y_true, novelty_matrix = _get_as_binary(results_frame)
    return np.apply_along_axis(lambda x: accuracy_score(y_true, x), 0, novelty_matrix)

def classf_accuracy_score(results_frame):
    """Computes classification accuracy for each threshold"""
    labels = results_frame.loc[:, 'Labels'].values.flatten()
    pred_matrix = results_frame.loc[:, 'Classification'].values
    classf_true = labels[labels != 'Nov']
    classf_pred = pred_matrix[labels != 'Nov']
    return np.apply_along_axis(lambda x: accuracy_score(classf_true, x), axis=0, arr=classf_pred)

def classf_accuracy_per_class(results_frame, class_=None):
    """Computes classification accuracy for each threshold for a specific class if a class is passed.
    If no class_ parameter is passed the accuracy is computed for each classed and returned in sorted order"""
    if class_ is None:
        classes = np.unique(results_frame.loc[:, 'Labels'].values.flatten())
        known_classes = classes[classes != 'Nov']
        acc = list()
        for class_ in known_classes:
            acc.append(_class_acc_per_class(results_frame, class_))
        return np.array(acc)
    else:
        return _class_acc_per_class(results_frame, class_)

def _class_acc_per_class(results_frame, class_):
    pred_matrix = results_frame.loc[:, 'Classification'].values
    labels = results_frame.loc[:, 'Labels'].values.flatten()
    class_labels = labels[labels == class_]
    class_pred = pred_matrix[labels == class_]
    return np.apply_along_axis(lambda x: accuracy_score(class_labels, x), axis=0, arr=class_pred)    

def noc_auc_score(results_frame):
    """Computes noc auc for given array of thresholds in a results_frame"""
    trigger, novelty_rate = get_recall_score(results_frame)
    noc_auc = math_utils.trapezoid_integration(novelty_rate, trigger)
    return noc_auc

def _get_as_binary(results_frame):
    """Gets novelty matrix from the frame as binary, 1 for novelty, 0 for known."""
    y_true = np.where(results_frame.loc[:, 'Labels'].values.flatten() == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:,'Classification'] == 'Nov', 1, 0)
    return y_true, novelty_matrix

def _get_novelty_matrix(predictions, threshold, neuron_names):
        """
        Returns a novelty detection matrix with the classification in the cases where novelty was
        not detected with the threshold array as columns and events as rows.
        It uses winner takeas all method for classification and a threshold on the output layer for 
        novelty detection. 

        Parameters

        predictions: numpy array
            Output from the output layer

        threshold: numpy array
            Array with the threshold values

        neuron_names: numpy.ndarray
            Array with the name of the class obtained from each neuron
        
        Returns:
            
        novelty_matrix: numpy.ndarray
            The classification and novelty detection of the model per threshold
        """

        novelty_matrix = list()
        for t in threshold:
            novelty_detection = (predictions < t).all(axis=1).flatten()
            classfication = neuron_names[np.argmax(predictions, axis=1)]
            novelty_matrix.append(np.where(novelty_detection, 'Nov', classfication))
        
        return np.column_stack(tuple(novelty_matrix))