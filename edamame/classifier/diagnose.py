#TODO - ROCAUC

import edamame.eda as eda
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Union
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import random

class ClassifierDiagnose:
    """
    A class for diagnosing classification models.

    Attributes:
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.

    Examples:
        >>> from edamame.classifier import TrainClassifier
        >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        >>> nb = classifier.gaussian_nb()
        >>> classifiers_diagnose = ClassifierDiagnose(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        >>> classifiers_diagnose.class_prediction_error(model=nb)
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def class_prediction_error(self, model: Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC] , train_data: bool = True) -> None:
        """
        This plot shows the support (number of training samples) for each class in the fitted classification model as a stacked bar chart. Each bar is segmented to show the proportion of predictions (including false negatives and false positives, like a Confusion Matrix) for each class

        Args:
            model (Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC]): Classification model.
            train_data (bool): Defines if you want to plot the stacked barplot on train or test data (train by default).
        
        Returns: 
            None
        """
        def stacked_barplot(matrix: np.array, num_of_class: int) -> None:
            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(8,6))
            cm_bottom = np.zeros((num_of_class,))
            for i in range(len(matrix)):
                value = random.uniform(0, 1)
                clr = cmap(value)
                if i == 0:
                    plt.bar(range(num_of_class), matrix[i], label=f'Class {i}', color=clr)
                else: 
                    cm_bottom += matrix[i-1]
                    plt.bar(range(num_of_class), matrix[i], bottom=cm_bottom, label=f'Class {i}', color=clr)
            plt.xticks(range(num_of_class), [str(i) for i in range(num_of_class)])
            plt.ylabel('Count')
            plt.title('Predicted vs True Class')
            plt.legend()
            plt.show()

        if train_data:
            y_pred = model.predict(self.X_train)
            y_true = self.y_train.squeeze().to_numpy() 
            num_classes = y_true.max() + 1
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            stacked_barplot(matrix=cm, num_of_class=num_classes)
        else:
            y_pred = model.predict(self.X_test)
            y_true = self.y_test.squeeze().to_numpy() 
            num_classes = y_true.max() + 1
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            stacked_barplot(matrix=cm, num_of_class=num_classes)