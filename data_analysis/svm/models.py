import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVC
from multiprocessing import Pool
from data_analysis.utils import utils

class OneVsRestSVCCommittee():
    """Implements a One vs Rest classificator using a SVM committee
    A SVC is trained for each class which has the value 1 while the non-class occurences have the value -1
    
    Attributes:
    
    class_mapping: dict
        Maps the value of a class key to its value
        
    classifiers: dict
        Each class key contains the SVC for the given class"""

    def __init__(self, class_mapping, expert_params=None):
        """
        Parameters:
        
        class_mapping: dict
            Maps the value of a class key to its value. The dict is sorted by key order.

        expert_params: dict
            Each class key must have a dict with the paramter mapping for the SVC of that class
        """
        class_mapping = utils.sort_dict(class_mapping)
        self.class_mapping = class_mapping
        if expert_params is None:
            expert_params = dict.fromkeys(list(class_mapping.keys()), dict())
        self.classifiers = {class_: SVC(**params) for class_, params in expert_params.items()}
        self.n_classes = len(self.class_mapping.items())

    def fit(self, X, y, n_workers=None, verbose=True):
        """Fits the SVC's for the given data

        Parameters:

        X: numpy.ndarray
            Training samples

        y: numpy.array
            True labels for X
        
        n_workers: int
            Number of processes to fit the svc's. If None is passed, it defaults to os.cpu_count()
        
        verbose: bool
            If True outputs which SVC is being trained
        """

        pool = Pool(processes=n_workers)

        if verbose:
            print('Starting the SVC fit')

        y_data = (y for _ in range(self.n_classes))
        x_data = (X for _ in range(self.n_classes))
        trained_models = pool.starmap(self._train_one_model, zip(self.class_mapping.values(), self.classifiers.values(), x_data, y_data), )
        del self.classifiers
        self.classifiers = {class_: classifier for class_, classifier in zip(self.class_mapping.keys(), trained_models)}
        
        if verbose:
            print('Finished training')

    def get_params(self):
        """Returns a dict which each class key has the params from the SVC of the given class"""
        return {class_: classifier.get_params() for class_, classifier in self.classifiers.items()}

    def predict(self, X):
        """Returns a ndarray of size (len(X), n_classes) where each column is the output of the
        SVC in sorted class name order"""
        return np.column_stack([classifier.predict(X) for classifier in self.classifiers.values()])
    
    def score(self, X, y):
        """Returns a ndarray (n_classes,) with the mean accuracy on the given test data and labels.
        Each value is the output of SVC.score in sorted class name order.
        
        Parameters:
        
        X: numpy.ndarray
            Test samples
            
        y: numpy.ndarray
            Labels for X
            
        Returns:
        
        all_scores: numpy.ndarray
            Array with the mean accuracy
        """
        stack = list()
        for class_name, class_value in self.class_mapping.items():
            stack.append(self.classifiers[class_name].score(X, np.where(y == class_value, 1, -1)))
        all_scores = np.column_stack(stack)
        del stack
        return all_scores   

    def evaluate(self, X, y, metrics):
        """Evaluates the given metrics for the given data in each SVC

        Parameters:

        X: numpy.ndarray
            Test samples

        y: numpy.ndarray
            Lbales for X

        metrics: list or iterable
            Iterable where each iteration returns a callable that computes a metric.It must take two 
            arguments in the given order: (y_true: true labels, y_pred: predicted labels) and returns 
            a float or int
        
        Returns:
        
        frame: pandas.DataFrame
            Frame with the computed metrics with each column being the class of the SVC and each line
            being a metric.
        """
        columns = np.array(self.class_mapping.keys())
        index = list()
        data = list()
        for metric in metrics:
            index.append(metric.__name__)
            data.append([metric(np.where(y == class_value, 1, -1), self.classifiers[class_name].predict(X))
                            for class_name, class_value in self.class_mapping.items()])
        
        frame = pd.DataFrame(data, index=index, columns=columns)
        return frame

    def set_params(self, class_, **params):
        """Sets the params of a SVC for a given class
        
        Parameters:
        
        class_:
            Class of the SVC to have its parameters set
        """
        self.classifiers[class_].set_params(**params)

    @staticmethod
    def _train_one_model(class_value, model, X, y):
        """Trains one model. Implemented for multiprocessing.Pool support"""
        return model.fit(X, np.where(y == class_value, 1, -1))