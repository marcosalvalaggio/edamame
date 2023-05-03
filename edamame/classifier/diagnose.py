#TODO - Class Prediction Error

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

class ClassifierDiagnose:
    """
    A class for diagnosing classification models.

    Attributes:
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def class_prediction_error(self, model: Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC]) -> None:
        # training dataset
        y_pred = model.predict(self.X_train)
        y_true = self.y_train.squeeze().to_numpy() 
