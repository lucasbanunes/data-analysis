import numpy as np
import pandas as pd
from data_analysis.utils import utils

def wta_classf_analysis(output, labels):
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

    Returns:
    frame: pandas.DataFrame
        Frame with the class as rows and the columns as the metrics
    """
    classes = np.unique(labels)
    classf = classes[np.argmax(output, axis=1)]
    data = list()
    for class_index in range(len(classes)):
        class_ = classes[class_index]
        unique_classf, unique_counts = np.unique(classf[labels == class_], return_counts=True)
        num_events = np.sum(unique_counts)
        num_correct_classf = unique_counts[unique_classf == class_][0]
        num_misclassf = np.sum(unique_counts[unique_classf != class_])
        num_most_misclassf = utils.second_max(unique_counts)
        most_misclassf_class = unique_classf[unique_counts == num_most_misclassf][0]
        class_neuron_output = output[labels == class_].T[class_index]
        avg = np.sum(class_neuron_output)/len(class_neuron_output)
        err = np.sqrt(np.var(class_neuron_output))
        max_value = max(class_neuron_output)
        min_value = min(class_neuron_output)
        acc = num_correct_classf/num_events
        data.append([num_events, num_correct_classf, avg, err, max_value, min_value, num_misclassf, most_misclassf_class, num_most_misclassf, acc])
        print(class_, unique_classf, unique_counts)
    return pd.DataFrame(data, index=classes, columns=['Events', 'Correct', 'Avg', 'Error', 'Max', 'Min', 'Misclassf', 'Most Misclassf', 'Most events', 'Acc'])
