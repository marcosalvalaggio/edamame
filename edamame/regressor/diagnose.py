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
from typing import Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def check_random_forest(model: RandomForestRegressor) -> None: 
    """
    The function checks if the model passed is a random forest regression.

    Args:
        model (RandomForestRegressor): The input model to be checked.

    Raises:
            TypeError: If the input model is not a random forest regression model.
    
    Returns:
        None
    """
    if model.__class__.__name__ != 'RandomForestRegressor':
        raise TypeError('The model passed isn\'t a ridge model')
    

def check_xgboost(model: xgb.XGBRegressor) -> None: 
    """
    The function checks if the model passed is a xgboost regression.

    Args:
        model (xgb.XGBRegressor): The input model to be checked.

    Raises:
        TypeError: If the input model is not an XGBoost regression model.

    Returns:
        None
    """
    if model.__class__.__name__ != 'XGBRegressor':
        raise TypeError('The model passed isn\'t an xgboost')


class Diagnose:
    """
    A class for diagnosing regression models.

    Attributes:
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.

    Methods:
        coefficients: Display coefficients for Linear, Lasso, and Ridge model.
        random_forest_fi: The function displays the feature importance plot. 
        xgboost_fi: The function displays the feature importance plot.
        prediction_error: Define a scatterpolot with ygt and ypred of the model passed.
        residual_plot:  Residual plot for train and test data.
        qqplot: QQplot for train and test data.

    Example:
        >>> from edamame.regressor import TrainRegressor, Diagnose
        >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
        >>> linear = regressor.linear()
        >>> diagnose = Diagnose(X_train, np.log(y_train), X_test, np.log(y_test))
        >>> diagnose.coefficients()
        >>> diagnose.prediction_error(model=linear)
        >>> diagnose.residual_plot(model=linear)
        >>> diagnose.qqplot(model=linear)
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """
        Initializes the Diagnose object with training and test data.

        Args:
            X_train (pd.DataFrame): The input training data.
            y_train (pd.Series): The target training data.
            X_test (pd.DataFrame): The input test data.
            y_test (pd.Series): The target test data.

        Returns:
            None
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def coefficients(self, model: Union[LinearRegression, Lasso, Ridge]) -> None:
        """
        Display coefficients for Linear, Lasso, and Ridge model.

        Args:
            model (Union[LinearRegression, Lasso, Ridge]): The input model for which coefficients need to be displayed.

        Returns:
            None
        """
        intercept = ('intercept',model.intercept_)
        coef = list(zip(list(model.feature_names_in_), model.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Var', 'Coeff.']
        # display step 
        display(df_coef)


    def random_forest_fi(self, model: RandomForestRegressor, figsize: Tuple[float, float] = (12,10)) -> None:
        """
        The function displays the feature importance plot. 

        Args:
            model (RandomForestRegressor): The input random forest model.

        Returns:
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


    def xgboost_fi(self, model: xgb.XGBRegressor, figsize: tuple[float, float] = (14,12)) -> None:
        """
        The function displays the feature importance plot.

        Args:
            model (xgb.XGBRegressor): The input xgboost model.

        Returns:
            None
        """
        check_xgboost(model)
        xgb.plot_importance(model)
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
        plt.show()

    
    def prediction_error(self, model: Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor], train_data: bool = True, figsize: Tuple[float, float] = (8.,6.)) -> None:
        """
        Define a scatterpolot with ygt and ypred of the model passed.

        Args:
            model (Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]): The input model.
            train (bool): Defines if you want to plot the scatterplot on train or test data (train by default).
            figsize (Tuple[float, float]): Define the size of the prediction_erros plot.

        Returns:
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


    def residual_plot(self, model: Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]) -> None:
        """
        Residual plot for train and test data.

        Args:
            model (Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]): The input model. 

        Returns:
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


    def qqplot(self, model: Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]) -> None:
        """
        QQplot for train and test data.

        Args:
            model (Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]): The input model. 

        Returns:
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
