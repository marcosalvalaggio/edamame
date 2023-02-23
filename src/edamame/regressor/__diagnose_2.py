import pandas as pd 
from IPython.display import display
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt 



def check_random_forest(model): 
    """
    Parameters:
    :model - Sklearn model 
    ---------------------------
    The function checks if the model passed is a random forest regression.
    """
    if model.__class__.__name__ != 'RandomForestRegressor':
        raise TypeError('The model passed isn\'t a ridge model')
    

def check_xgboost(model): 
    """
    Parameters:
    :model - xgboost model
    ---------------------------
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
        Display model coefficients 
        """
        intercept = ('intercept',model.intercept_)
        coef = list(zip(list(model.feature_names_in_), model.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Var', 'Coeff.']
        # display step 
        display(df_coef)

    def random_forest_fi(model, figsize: tuple[float, float] = (12,10)):
        """
        Parameters:
        :model - A RF regression model
        ---------------------------
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


    def xgboost_fi(model, figsize: tuple[float, float] = (12,10)):
        """
        Parameters:
        :model - A xgboost regression model
        ---------------------------
        The function displays the feature importance plot. 
        """
        check_xgboost(model)
        xgb.plot_importance(model)
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
        plt.show()
