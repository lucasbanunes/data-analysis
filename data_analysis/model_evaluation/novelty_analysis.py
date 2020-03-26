import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score

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
                save_csv = False, 
                filepath = './',
                filename = None):
        """
        Creates a pandas.DataFrame with the events as rows and the output of the neurons in the output layer, 
        the classification for each threshold and the labels as columns.
        It uses winner takeas all method for classification and a threshold on the output layer for 
        novelty detection.

        Parameters

        predictions: numpy array
            Output from the output layer

        threshold: numpy array
            Array with the threshold values

        classes_names: 1-d array like
            Array with the name of the class obtained from each neuron

        novelty_index: int
            Number of the label that is treated as novelty

        save_csv: boolean
            If true saves the data frame as a .csv file

        filepath: string
            Where to save the .csv file

        filename: string
            Name of the file to be saved

        Returns:

        novelty_frame: pandas.DataFrame
            Frame with all the data with the following organization:
            Events as rows
            Two layers of columns:
                Neurons: predictions data where each subcolumn is named out0, out1, etc
                Classification: the model classification with the threshold as subcolumns
                Labels: with one subcolumn L with the correct labels of the data set
        """

        classes_names = np.array(classes_names)
        novelty_matrix = _get_novelty_matrix(predictions, threshold, classes_names)
        labels_matrix = np.insert(classes_names, novelty_index, 'Nov')[labels].reshape(-1,1)
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
        if save_csv:
            filepath = os.path.join(filepath, filename)
            novelty_frame.to_csv(filepath, index = False)
        return novelty_frame

def get_novelty_eff(results_frame,
                    sample_weight=None, 
                    save_csv = False,
                    filepath = './',
                    filename = None,
                    verbose = False):
    """
    returns a data frame with several parameters of efficiency for novelty detection

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results function

    save _csv: boolean
        if true saves the data frame as a .csv file

    filepath: string
        where to save the .csv file
    
    filename: string
        name of the file to be saved
    
    verbose: boolean
        if true gives output of the function's activity
    
    Return
        eff_frame: pandas.DataFrame
    """
    y_true = np.where(results_frame.loc[:, 'Labels'].values == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:'Classification'] == 'Nov', 1, 0)
    threshold = results_frame.loc[:, 'Classification'].columns.values.flatten()
    recall = np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None, sample_weight=sample_weight), 0, novelty_matrix)
    precision = np.apply_along_axis(lambda x: precision_score(y_true, x, labels = [0,1], average=None, sample_weight=sample_weight), 0, novelty_matrix)
    nov_eff_frame = pd.DataFrame(np.vstack((recall, precision)), columns = ['Recall', 'Precision'], index=pd.MuliIndex.from_product([['Known', 'Nov'], threshold], names=('Class', 'Threshold')))
    if save_csv:
        filepath = os.path.join(filepath, filename)
        nov_eff_frame.to_csv(filepath, index = True)

    return nov_eff_frame    

def plot_noc_curve(results_frame, nov_class_name, figsize=(12,3), save_fig = True, filepath=''):
    y_true = np.where(results_frame.loc[:, 'Labels'].values == 'Nov', 1, 0)     #Novelty is 1, known data is 0
    novelty_matrix = np.where(results_frame.loc[:'Classification'] == 'Nov', 1, 0)
    recall = np.apply_along_axis(lambda x: recall_score(y_true, x, labels=[0,1], average=None), 0, novelty_matrix)
    fig = plt.figure(figsize=(12,3))
    axis = fig.add_axes([0.1,0.1,0.8,0.8])
    axis.set_title(f'NOC Curve Novelty class{nov_class_name}')
    axis.set_ylabel('Trigger Rate')
    axis.set_ylim(0,1)
    axis.set_xlim(0,1)
    axis.set_xlabel('Novelty Rate')
    axis.plot(recall[1], recall[0])
    if save_fig:
        fig.savefig(filepath, dpi=200, format='png')
    plt.close(fig)
    return fig
                    
def _get_novelty_matrix(predictions, threshold, classes_names):
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

        classes_names: numpy.ndarray
            Array with the name of the class obtained from each neuron
        
        Returns:
            
        novelty_matrix: numpy.ndarray
            The classification and novelty detection of the model per threshold
        """

        novelty_matrix = list()
        for t in threshold:
            novelty_detection = (predictions < t).all(axis=1).flatten()
            classfication = classes_names[np.argmax(predictions, axis=1)]
            novelty_matrix.append(np.where(novelty_detection, 'Nov', classfication))
        
        return np.column_stack(tuple(novelty_matrix))