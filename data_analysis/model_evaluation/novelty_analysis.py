import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score
from data_analysis.utils.utils import NumericalIntegration

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

def get_novelty_eff(results_frame,
                    sample_weight=None, 
                    filepath = None):
    """
    returns a data frame with several parameters of efficiency for novelty detection

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results function
    
    sample_weight: list
        Sample weights for recall and precision

    filepath: string
        Where to save the .csv file. Defaults to None and if that happens the frame is not saved.
    
    Return
        eff_frame: pandas.DataFrame
    """
    y_true = np.where(results_frame.loc[:, 'Labels'].values == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:'Classification'] == 'Nov', 1, 0)
    threshold = results_frame.loc[:, 'Classification'].columns.values.flatten()
    recall = np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None, sample_weight=sample_weight), 0, novelty_matrix)
    precision = np.apply_along_axis(lambda x: precision_score(y_true, x, labels = [0,1], average=None, sample_weight=sample_weight), 0, novelty_matrix)
    nov_eff_frame = pd.DataFrame(np.vstack((recall, precision)), columns = ['Recall', 'Precision'], index=pd.MuliIndex.from_product([['Known', 'Nov'], threshold], names=('Class', 'Threshold')))
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        nov_eff_frame.to_csv(filepath, index = False)
    return nov_eff_frame    

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
        noc_area = NumericalIntegration.trapezoid_rule(novelty_rate, trigger)
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
    """Plot noc curve fro given results_frame

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
    y_true = np.where(results_frame.loc[:, 'Labels'].values.flatten() == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:,'Classification'] == 'Nov', 1, 0)
    threshold = results_frame.loc[:,'Classification'].columns.values.flatten()
    acc = np.apply_along_axis(lambda x: accuracy_score(y_true, x), 0, novelty_matrix)
    fig, axis = plt.subplots(figsize=figsize)
    axis.set_title(f'Novelty class {nov_class_name}')
    axis.set_ylabel('Accuracy')
    axis.set_ylim(0,1)
    axis.set_xlabel('Threshold')
    axis.plot(threshold, acc, color='k')
    axis.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5)
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
    noc_auc = NumericalIntegration.trapezoid_rule(novelty_rate, trigger)
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