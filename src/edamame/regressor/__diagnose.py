#TODO - aggiungere al residual_plot l'opzione per il qqplot 
#TODO - aggiungere se fattibili il plot per la cook distance

import pandas as pd 
import numpy as np
from IPython.display import display
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats


def check_random_forest(model) -> None: 
    """
    The function checks if the model passed is a random forest regression.

    Parameters
    ----------
    model:
        A Sklearn model 

    Raises
    ----------
    TypeError
        Control the model type
    
    Returns
    ----------
    None
    """
    if model.__class__.__name__ != 'RandomForestRegressor':
        raise TypeError('The model passed isn\'t a ridge model')
    

def check_xgboost(model): 
    """
    The function checks if the model passed is a xgboost regression.

    Parameters
    ----------
    model: 
        A xgboost model

    Raises
    ----------
    TypeError
        Control the model type

    Returns
    ----------
    None
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

        Parameters
        ----------
        model: 
            A RF regression model

        Returns
        ----------
        None
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
        The function displays the feature importance plot. 

        Parameters
        ----------
        model: 
            A RF regression model

        Returns
        ----------
        None
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
        The function displays the feature importance plot.

        Parameters
        ----------
        model: 
            A xgboost regression model

        Returns
        ----------
        None 
        """
        check_xgboost(model)
        xgb.plot_importance(model)
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
        plt.show()

    
    def prediction_error(self, model, train_data: bool = True, figsize: tuple[float, float] = (8,6)):
        """
        Define a scatterpolot with ygt and ypred of the model passed.

        Parameters
        ----------
        model: 
            A model to analyze. 
        train: bool
            Defines if you want to plot the scatterplot on train or test data (train by default).
        figsize: tuple
            Define the size of the prediction_erros plot   

        Returns
        ----------
        None
        """
        if train_data: 
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

    def residual_plot(self, model):
        """
        Residual plot for train and test data

        Parameters
        ----------
        model: 
            A model to analyze. 

        Returns 
        ----------
        None
        """
        ypred_train = model.predict(self.X_train)
        ypred_test = model.predict(self.X_test)
        res_train = self.y_train.squeeze().to_numpy() - ypred_train
        res_test = self.y_test.squeeze().to_numpy() - ypred_test
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]}, figsize=(8, 6))
        sns.scatterplot(x=ypred_train, y=res_train, alpha=0.5, ax=ax1)
        sns.scatterplot(x=ypred_test, y=res_test, color='red', alpha=0.5, ax=ax1)
        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residual Plot')
        sns.histplot(y=res_train, alpha=0.5, ax=ax2)
        sns.histplot(y=res_test, color='red', alpha=0.5, ax=ax2)
        ax2.axhline(y=0, color='black', linestyle='--')
        ax2.set_xlabel('Distribution')
        plt.show()


    def qqplot(self, model):
        """
        QQplot for train and test data

        Parameters
        ----------
        model: 
            A model to analyze. 

        Returns
        ----------
        None
        """
        ypred_train = model.predict(self.X_train)
        ypred_test = model.predict(self.X_test)
        res_train = self.y_train.squeeze().to_numpy() - ypred_train
        res_test = self.y_test.squeeze().to_numpy() - ypred_test
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6))
        stats.probplot(res_train, plot=ax1)
        ax1.set_title('Train Residuals QQ Plot')
        stats.probplot(res_test, plot=ax2)
        ax2.set_title('Test Residuals QQ Plot')
        fig.tight_layout()
        plt.show()
