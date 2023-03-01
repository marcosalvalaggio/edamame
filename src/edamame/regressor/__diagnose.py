#TODO - creare funzione prediction_error e residual_plot
#TODO - modificare prediction_error aggiungendo legenda su train e test data

import pandas as pd 
import numpy as np
from IPython.display import display
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import seaborn as sns


def check_random_forest(model) -> None: 
    """
    Parameters
    ----------
    model:
        A Sklearn model 
    
    Return
    ----------
    None
        The function checks if the model passed is a random forest regression.
    """
    if model.__class__.__name__ != 'RandomForestRegressor':
        raise TypeError('The model passed isn\'t a ridge model')
    

def check_xgboost(model): 
    """
    Parameters
    ----------
    model: 
        A xgboost model

    Return
    ----------
    None
        The function checks if the model passed is a xgboost regression.
    """
    if model.__class__.__name__ != 'XGBRegressor':
        raise TypeError('The model passed isn\'t an xgboost')


class Diagnose:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def coefficients(self, model):
        """
        Parameters
        ----------
        model: 
            A RF regression model

        Return
        ----------
        None
            Display model coefficients 
        """
        intercept = ('intercept',model.intercept_)
        coef = list(zip(list(model.feature_names_in_), model.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Var', 'Coeff.']
        # display step 
        display(df_coef)

    def random_forest_fi(self, model, figsize: tuple[float, float] = (12,10)):
        """
        Parameters
        ----------
        model: 
            A RF regression model

        Return
        ----------
        None
            The function displays the feature importance plot. 
        """
        check_random_forest(model)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        feature_names = model.feature_names_in_
        forest_importances = pd.Series(importances, index=feature_names)
        plt.figure(figsize=figsize)
        forest_importances.plot.bar(yerr=std)
        plt.title("Feature importances using mean decrease in impurity")
        plt.ylabel("Mean decrease in impurity")
        plt.show()


    def xgboost_fi(self, model, figsize: tuple[float, float] = (14,12)):
        """
        Parameters
        ----------
        model: 
            A xgboost regression model

        Return
        ----------
        None
            The function displays the feature importance plot. 
        """
        check_xgboost(model)
        xgb.plot_importance(model)
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
        plt.show()

    
    def prediction_error(self, model, train: bool = True, figsize: tuple[float, float] = (8,6)):
        """
        Parameters
        ----------
        model: 
            A regression model to analyze. 
        train: bool
            Defines if you want to plot the scatterplot on train or test data (train by default).

        Return 
        ----------
        None
            Define a scatterpolot with ygt and ypred of the model passed.
        """
        if train: 
            ypred = model.predict(self.X_train)
            ygt = self.y_train.squeeze().to_numpy()
            r2 = r2_score(self.y_train, ypred)
            df = pd.DataFrame({"ypred": ypred, "y": ygt})
            # scatterplot 
            plt.figure(figsize=figsize)
            sns.scatterplot(data=df, x="y", y="ypred")
            plt.annotate(f"R2 train: {r2:.4f}", xy = (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12)
            plt.show()
        else: 
            ypred = model.predict(self.X_test)
            ygt = self.y_test.squeeze().to_numpy()
            r2 = r2_score(self.y_test, ypred)
            df = pd.DataFrame({"ypred": ypred, "y": ygt})
            # scatterplot 
            plt.figure(figsize=figsize)
            sns.scatterplot(data=df, x="y", y="ypred")
            plt.annotate(f"R2 test: {r2:.4f}", xy = (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=12)
            plt.show()

