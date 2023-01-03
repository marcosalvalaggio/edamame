import numpy as np
import pandas as pd
from tools import dataframe_review, dummy_control
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class TrainClassifier:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # check dataframe 
        dataframe_review(self.X_train)
        dataframe_review(self.X_test)
        # check columns type
        dummy_control(self.X_train)
        dummy_control(self.X_test)

    # ------------ #
    # logistic model
    # ------------ #
    def logistic(self):
        logistic = LogisticRegression()
        logistic.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.logistic = logistic
        # return step 
        return logistic


    # ------------ #
    # Naive gaussian Bayes
    # ------------ #
    def gaussian_nb(self):
        gauss_nb = GaussianNB()
        gauss_nb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.gauss_nb = gauss_nb
        # return step 
        return gauss_nb