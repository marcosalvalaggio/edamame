#TODO - add parameteres "verbose" for logging message like unable to print/save

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from IPython.display import display, Markdown
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, get_scorer_names
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, cross_val_score
import pickle
from edamame.eda.tools import dataframe_review, dummy_control, setup
from typing import Tuple, List, Literal, Union
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# list of metrics
#get_scorer_names()


class TrainRegressor:
    """
    This class represents a pipeline for training and handling regression models.

    Attributes: 
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.

    Example: 
        >>> from edamame.regressor import TrainRegressor
        >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
        >>> linear = regressor.linear()
        >>> regressor.model_metrics(model_name="linear")
        >>> regressor.save_model(model_name="linear")
        >>> lasso = regressor.lasso()
        >>> ridge = regressor.ridge()
        >>> tree = regressor.tree()
        >>> rf = regressor.random_forest()
        >>> xgb = regressor.xgboost()
        >>> regressor.model_metrics()
        >>> # using AutoML
        >>> models = regressor.auto_ml()
        >>> regressor.model_metrics()
        >>> regressor.save_model()
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test, y_test):
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
        # models
        self.__linear_fit = {}
        self.__lasso_fit = {}
        self.__ridge_fit = {}
        self.__tree_fit = {}
        self.__random_forest_fit = {}
        self.__xgb_fit = {}


    def linear(self) -> LinearRegression:
        """
        Train a linear regression model using the training data and return the fitted model.

        Returns:
            LinearRegression: The trained linear regression model.
        
        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> linear = regressor.linear()
        """
        linear = LinearRegression()
        linear.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__linear_fit = linear
        # return step 
        return self.__linear_fit


    def lasso(self, alpha: Tuple[float, float, int] = (0.0001, 10., 50), n_folds: int = 5) -> Lasso:
        """
        Train a Lasso regression model using the training data and return the fitted model.

        Args:
            alpha (Tuple[float, float, int]): The range of alpha values to test for hyperparameter tuning. Default is (0.0001, 10., 50).
            n_folds (int): The number of cross-validation folds to use for hyperparameter tuning. Default is 5.

        Returns:
            Lasso: The trained Lasso regression model.

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> lasso = regressor.lasso(alpha=(0.0001, 10., 50), n_folds=5)
        """
        # lasso hyperparameter 
        alphas = np.linspace(alpha[0], alpha[1], alpha[2])
        # hyperparameter gridsearch
        lasso = Lasso()
        tuned_parameters = [{"alpha": alphas}]
        reg_lasso = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_lasso.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__lasso_fit = reg_lasso.best_estimator_
        # return step 
        return self.__lasso_fit


    def ridge(self, alpha: Tuple[float, float, int] = (0.1, 50., 50), n_folds: int = 5) -> Ridge:
        """
        Train a Ridge regression model using the training data and return the fitted model.

        Args:
            alpha (Tuple[float, float, int]): The range of alpha values to test for hyperparameter tuning. Default is (0.1, 50, 50).
            n_folds (int): The number of cross-validation folds to use for hyperparameter tuning. Default is 5.

        Returns:
            Ridge: The trained Ridge regression model.

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> ridge = regressor.ridge(alpha=((0.1, 50., 50), n_folds=5)
        """
        # ridge hyperparameter 
        alphas = np.linspace(alpha[0], alpha[1], alpha[2])
        # hyperparameter gridsearch
        ridge = Ridge()
        tuned_parameters = [{"alpha": alphas}]
        reg_ridge = GridSearchCV(ridge, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_ridge.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__ridge_fit = reg_ridge.best_estimator_
        # return step 
        return self.__ridge_fit
    
    
    def tree(self, alpha: Tuple[float, float, int] = (0., 0.001, 5), impurity: Tuple[float, float, int] = (0., 0.00001, 5),
            n_folds: int = 5) -> DecisionTreeRegressor:
        """
        Fits a decision tree regression model using the provided training data and hyperparameters.

        Args:
            alpha (Tuple[float, float, int]): A tuple specifying the range of values to use for the ccp_alpha 
                hyperparameter. The range is given as a tuple (start, stop, num), where `start` is the start
                of the range, `stop` is the end of the range, and `num` is the number of values to generate
                within the range. Defaults to (0., 0.001, 5).
            impurity (Tuple[float, float, int]): A tuple specifying the range of values to use for the 
                min_impurity_decrease hyperparameter. The range is given as a tuple (start, stop, num), where 
                `start` is the start of the range, `stop` is the end of the range, and `num` is the number of 
                values to generate within the range. Defaults to (0., 0.00001, 5).
            n_folds (int): The number of folds to use for cross-validation. Defaults to 5.

        Returns:
            DecisionTreeRegressor: The fitted decision tree regressor model.

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> tree = regressor.tree(alpha=(0., 0.001, 5), impurity=(0., 0.00001, 5), n_folds=3)
        """
        # hyperparameters gridsearch
        alphas = np.linspace(alpha[0], alpha[1], alpha[2])
        impurities = np.linspace(impurity[0], impurity[1], impurity[2])
        tuned_parameters = [{"ccp_alpha": alphas, 'min_impurity_decrease': impurities}]
        tree = DecisionTreeRegressor() 
        reg_tree = GridSearchCV(tree, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_tree.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__tree_fit = reg_tree.best_estimator_
        # return step 
        return self.__tree_fit


    def random_forest(self, n_estimators: Tuple[int, int, int] = (50, 1000, 5), n_folds: int = 2) -> RandomForestRegressor:
        """
        Trains a Random Forest regression model on the training data and returns the best estimator found by GridSearchCV.

        Args:
            n_estimators (Tuple[int, int, int]): A tuple of integers specifying the minimum and maximum number of trees
                to include in the forest, and the step size between them.
            n_folds (int): The number of cross-validation folds to use when evaluating models.

        Returns:
            RandomForestRegressor: The best Random Forest model found by GridSearchCV.
        """
        n_estimators = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = [{"n_estimators": n_estimators}]
        random_forest = RandomForestRegressor(warm_start=True, n_jobs=-1)
        reg_random_forest = GridSearchCV(random_forest, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_random_forest.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__random_forest_fit = reg_random_forest.best_estimator_
        # return step 
        return self.__random_forest_fit


    def xgboost(self, n_estimators: Tuple[int, int, int] = (10, 100, 5), n_folds: int = 2) -> xgb.XGBRegressor:
        """
        Trains an XGBoost model using the specified hyperparameters.

        Args:
            n_estimators (Tuple[int, int, int]): A tuple containing the start, end and step values for number of estimators.
                Default is (10, 100, 5).
            n_folds (int): The number of folds to use in the cross-validation process. Default is 2.

        Returns:
            xgb.XGBRegressor: The trained XGBoost model.

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> xgboost = regressor.xgboost(n_estimators=(10, 200, 10), n_folds=5)
        """
        n_est = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = {"n_estimators": n_est}
        xgb_m = xgb.XGBRegressor(objective ='reg:squarederror')
        reg_xgb = GridSearchCV(xgb_m, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_xgb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__xgb_fit = reg_xgb.best_estimator_
        # return step 
        return self.__xgb_fit


    def model_metrics(self, model_name: Literal["all", "linear", "lasso", "ridge", "tree", "random_forest", "xgboost"] = 'all') -> None:
        """
        Displays the metrics of a trained regression model. The metrics displayed are R2, MSE, and MAE for both the training
        and test sets.

        Args:
            model_name (Literal["all", "linear", "lasso", "ridge", "tree", "random_forest", "xgboost"]): The name of the model to display metrics for. Can be one of 'all', 'linear', 'lasso', 'ridge', 'tree',
                        'random_forest', or 'xgboost'. Defaults to 'all'.

        Returns:
            None
        
        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> xgboost = regressor.xgboost(n_estimators=(10, 200, 10), n_folds=5)
            >>> regressor.model_metrics(model_name="xgboost")
        """
        model_dct = {'linear': 0, 'lasso': 1, 'ridge': 2, 'tree': 3, 'random_forest': 4, 'xgboost': 5}
        model_list = [self.__linear_fit, self.__lasso_fit, self.__ridge_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'dict':
                        display(f'unable to show {key} model metrics')
                else:
                    y_pred_train = model_list[model_dct[key]].predict(self.X_train)
                    y_pred_test = model_list[model_dct[key]].predict(self.X_test)
                    # r2
                    r2_train = r2_score(self.y_train, y_pred_train)
                    r2_test = r2_score(self.y_test, y_pred_test)
                    # MSE
                    mse_train = mean_squared_error(self.y_train, y_pred_train)
                    mse_test = mean_squared_error(self.y_test, y_pred_test)
                    # MAE
                    mae_train = mean_absolute_error(self.y_train, y_pred_train)
                    mae_test = mean_absolute_error(self.y_test, y_pred_test)
                    # display step 
                    index_label = ['R2', 'MSE', 'MAE']
                    metrics = pd.DataFrame([[r2_train, r2_test], [mse_train, mse_test], [mae_train, mae_test]], index = index_label)
                    metrics.columns = [f'Train', 'Test']
                    string = f'### {key} model metrics:'
                    display(Markdown(string))
                    display(metrics)
        else:
            if model_list[model_dct[model_name]].__class__.__name__ == 'dict':
                display(f'unable to show {model_name} model metrics')
            else:
                y_pred_train = model_list[model_dct[model_name]].predict(self.X_train)
                y_pred_test = model_list[model_dct[model_name]].predict(self.X_test)
                # r2
                r2_train = r2_score(self.y_train, y_pred_train)
                r2_test = r2_score(self.y_test, y_pred_test)
                # MSE
                mse_train = mean_squared_error(self.y_train, y_pred_train)
                mse_test = mean_squared_error(self.y_test, y_pred_test)
                # MAE
                mae_train = mean_absolute_error(self.y_train, y_pred_train)
                mae_test = mean_absolute_error(self.y_test, y_pred_test)
                # display step 
                index_label = ['R2', 'MSE', 'MAE']
                metrics = pd.DataFrame([[r2_train, r2_test], [mse_train, mse_test], [mae_train, mae_test]], index = index_label)
                metrics.columns = [f'Train', 'Test']
                string = f'### {model_name} model metrics:'
                display(Markdown(string))
                display(metrics)


    def auto_ml(self, n_folds: int = 5, data: Literal['train', 'test'] = 'train') -> List:
        """
        Perform automated machine learning with cross validation on a list of regression models.
        
        Args:
            n_folds (int): Number of cross-validation folds. Defaults to 5.
            data (Literal['train', 'test']): Target dataset for cross-validation. 
                Must be either 'train' or 'test'. Defaults to 'train'.
        
        Returns:
            List: List of best-fit regression models for each algorithm.

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> model_list = regressor.auto_ml()
        """
        kfold = KFold(n_splits=n_folds)
        cv_mean = []
        score = []
        std = []
        regressor = ["Linear", "Lasso", "Ridge", "Tree", "Random Forest", "Xgboost"]
        try:
            model_list = [LinearRegression(), Lasso(alpha = self.__lasso_fit.alpha),
                          Ridge(alpha = self.__ridge_fit.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.__tree_fit.ccp_alpha, min_impurity_decrease=self.__tree_fit.min_impurity_decrease),
                          RandomForestRegressor(n_estimators = self.__random_forest_fit.n_estimators, warm_start=True, n_jobs=-1), 
                          xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = self.__xgb_fit.n_estimators)]
        except:
            # find best hyperparameters
            self.linear()
            self.lasso()
            self.ridge()
            self.tree()
            self.random_forest()
            self.xgboost()
            # model list 
            model_list = [LinearRegression(), Lasso(alpha = self.__lasso_fit.alpha),
                          Ridge(alpha = self.__ridge_fit.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.__tree_fit.ccp_alpha, min_impurity_decrease=self.__tree_fit.min_impurity_decrease),
                          RandomForestRegressor(n_estimators = self.__random_forest_fit.n_estimators, warm_start=True, n_jobs=-1),
                          xgb.XGBRegressor(objective ='reg:squarederror', n_estimators = self.__xgb_fit.n_estimators)]
        # cross validation loop 
        for model in model_list:
            if data == 'train':
                cv_result = cross_val_score(model, self.X_train, self.y_train.squeeze(), cv=kfold, scoring="r2")
            elif data == 'test':
                cv_result = cross_val_score(model, self.X_test, self.y_test.squeeze(), cv=kfold, scoring="r2")
            else:
                raise ValueError('insert valid target dataset (\'train\' or \'test\')')
            cv_mean.append(cv_result.mean())
            std.append(cv_result.std())
            score.append(cv_result)
        # dataframe for results 
        df_kfold_result = pd.DataFrame({"CV Mean": cv_mean, "Std": std}, index=regressor)
        # display step 
        string = f'### Metrics results on {data} set:'
        display(Markdown(string))
        display(df_kfold_result)
        # boxplot on R2
        box = pd.DataFrame(score, index=regressor)
        plt.figure(figsize=(10,8))
        box.T.boxplot()
        plt.show()

        return [self.__linear_fit, self.__lasso_fit, self.__ridge_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit]


    def save_model(self, model_name: Literal["all", "linear", "lasso", "ridge", "tree", "random_forest", "xgboost"] = 'all') -> None:
        """
        Saves the specified machine learning model or all models in the instance to a pickle file.

        Args:
            model_name (Literal["all", "linear", "lasso", "ridge", "tree", "random_forest", "xgboost"]): 
                The name of the model to save. Defaults to 'all'.
            
        Returns:
            None

        Example:
            >>> from edamame.regressor import TrainRegressor
            >>> regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
            >>> model_list = regressor.auto_ml()
            >>> regressor.save_model(model_name="all")
        """
        model_dct = {'linear': 0, 'lasso': 1, 'ridge': 2, 'tree': 3, 'random_forest': 4, 'xgboost': 5}
        model_list = [self.__linear_fit, self.__lasso_fit, self.__ridge_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'dict':
                        display(f'unable to save {key} model')
                else:
                    filename = f'{key}.pkl'
                    with open(filename, 'wb') as file:
                        pickle.dump(model_list[model_dct[key]], file)
                        display(f'{filename} saved')
        else:
            if model_list[model_dct[model_name]].__class__.__name__ == 'dict':
                display(f'unable to save {model_name} model')
            else:
                filename = f'{model_name}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(model_list[model_dct[model_name]], file)



def regression_metrics(model: Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor], X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Compute and display the regression metrics R2, MSE and MAE of the input model.
    
        Args:
            model (Union[LinearRegression, Lasso, Ridge, DecisionTreeRegressor, RandomForestRegressor, xgb.XGBRegressor]): Regression model.
            X (pd.DataFrame): Input features.
            y (pd.DataFrame): Target feature.
        
        Returns:
            None
        """
        # dataframe check
        dataframe_review(X)
        dummy_control(X)
        # pred step 
        y_pred = model.predict(X)
        # r2
        r2 = r2_score(y, y_pred)
        # MSE
        mse = mean_squared_error(y, y_pred)
        # MAE
        mae = mean_absolute_error(y, y_pred)
        # display step 
        index_label = ['R2', 'MSE', 'MAE']
        metrics = pd.DataFrame([r2,mse,mae], index = index_label)
        metrics.columns = ['Values']
        string = '### Model metrics:'
        display(Markdown(string))
        display(metrics)


if __name__ == '__main__':
    X = pd.read_csv('/Users/marcosalvalaggio/code/python/ds/data/melb_data/X.csv', sep = ';')
    y = pd.read_csv('/Users/marcosalvalaggio/code/python/ds/data/melb_data/y.csv', sep = ';')
    X_train, X_test, y_train, y_test = setup(X, y)
    regressor = TrainRegressor(X_train, np.log(y_train), X_test, np.log(y_test))
    regressor.linear()
    model_list = regressor.auto_ml()
    regressor.model_metrics()
    regressor.save_model()