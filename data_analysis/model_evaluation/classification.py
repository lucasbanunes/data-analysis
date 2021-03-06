import os
import numpy as np
import pandas as pd
from data_analysis.utils import utils

def wta_classf_analysis(output, labels, classes = None, filepath = None, verbose=False):
    """Function to analise the classification and neuron output fromn a neural network classifier
    using winner takes all method (wta). It is made considering the alphabetical or numerical order of the labels.
    This means that the first neuron will be the first class in sorted order and so forth. 
    This function builds a frame that evaluates for each class, the average with error of the neuron output for that class,
    with which class it was most misclassified and other metrics.

    Parameters:

    output: numpy.ndarray
        2D array with each row being the output array from the network

    labels: numpy.ndarray
        Target classification
    
    classes: numpy.ndarray
        Array where index i is the class from neuron i. If nothing is passed the classes are obtained from numpy.unique(labels)

    Returns:
    frame: pandas.DataFrame
        Frame with the class as rows and the columns as the metrics
    """
    if classes is None:
        classes = np.unique(labels)
    classf = classes[np.argmax(output, axis=1)]
    data = list()
    for class_index in range(len(classes)):
        class_ = classes[class_index]
        unique_classf, unique_counts = np.unique(classf[labels == class_], return_counts=True)
        num_events = np.sum(unique_counts)
        num_correct_classf = unique_counts[unique_classf == class_][0]
        num_misclassf = np.sum(unique_counts[unique_classf != class_])
        if len(unique_counts) > 1:
            num_most_misclassf = unique_counts[utils.sec_argmax(unique_counts)]
            most_misclassf_class = unique_classf[unique_counts == num_most_misclassf][0]
        else:
            num_most_misclassf = 0
            most_misclassf_class = None
        class_neuron_output = output[labels == class_].T[class_index]
        avg = np.sum(class_neuron_output)/len(class_neuron_output)
        err = np.sqrt(np.var(class_neuron_output))
        max_value = max(class_neuron_output)
        min_value = min(class_neuron_output)
        acc = num_correct_classf/num_events
        data.append([num_events, num_correct_classf, avg, err, max_value, min_value, num_misclassf, most_misclassf_class, num_most_misclassf, acc])
        if verbose:
            print(class_, unique_classf, unique_counts)
    frame = pd.DataFrame(data, index=classes, columns=['Events', 'Correct', 'Avg', 'Error', 'Max', 'Min', 'Misclassf', 'Most Misclassf', 'Most events', 'Acc'])
    if not filepath is None:
        folder, _ = os.path.split(filepath)
        if not os.path.exists(folder):
            os.makedirs(folder)
        frame.to_csv(filepath)
    return frame