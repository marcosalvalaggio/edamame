import numpy as np
import pandas as pd
from tools import dataframe_review, dummy_control
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from IPython.display import display, Markdown
import matplotlib.pyplot as plt 
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
        self.logistic_fit = logistic
        # return step 
        return self.logistic_fit


    # ------------ #
    # Naive gaussian Bayes
    # ------------ #
    def gaussian_nb(self):
        gauss_nb = GaussianNB()
        gauss_nb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.gaussian_nb_fit = gauss_nb
        # return step 
        return self.gaussian_nb_fit

    
    # ------------ #
    # model metrics
    # ------------ #
    def model_metrics(self, model_name: str = 'all'):
        model_dct = {'logistic_fit': 0, 'guassian_nb_fit': 1}
        model_list = [self.logistic_fit, self.gaussian_nb_fit]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'method':
                        display(f'unable to show {key} model metrics')
                else:
                    model_str = key 
                    model_str = model_str.replace("_fit", "")
                    model_str = model_str.upper()
                    model_str = model_str.replace("_", " ")
                    title = f'### {model_str} model metrics:'
                    display(Markdown(title))
                    y_pred_train = model_list[model_dct[key]].predict(self.X_train)
                    y_pred_test = model_list[model_dct[key]].predict(self.X_test)
                    plt.figure(figsize=(10,4))
                    plt.subplot(121)
                    sns.heatmap(confusion_matrix(self.y_train, y_pred_train), annot=True, fmt="2.0f")
                    plt.title(f'{model_str} train')
                    plt.subplot(122)
                    sns.heatmap(confusion_matrix(self.y_test, y_pred_test), annot=True, fmt="2.0f")
                    plt.title(f'{model_str} test')
                    plt.show()
                    print(classification_report(self.y_train, y_pred_train))
                    print(classification_report(self.y_test, y_pred_test))

    