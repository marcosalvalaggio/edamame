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
from edamame.tools import dataframe_review, dummy_control, setup
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# list of metrics
#get_scorer_names()

# ----------------- #
# DUBBI/RIFLESSIONI 
# riflettere se mettere i modelli come membri privati/protetti della classe 
# ----------------- #

# ----------------- #
# REGRESSOR CLASS
# ----------------- #
class TrainRegressor:
    
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
        # models
        self.__linear_fit = {}
        self.__lasso_fit = {}
        self.__ridge_fit = {}
        self.__tree_fit = {}
        self.__random_forest_fit = {}
        self.__xgb_fit = {}


    # ------------ #
    # linear model
    # ------------ #
    def linear(self):
        linear = LinearRegression()
        linear.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__linear_fit = linear
        # return step 
        return self.__linear_fit


    # ------------ #
    # Lasso model
    # ------------ #
    def lasso(self, alpha: list[float, float, int] = [0.0001, 10., 50],  n_folds: int = 5):
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


    # ------------ #
    # Ridge model
    # ------------ #
    def ridge(self, alpha: list[float, float, int] = [0.1, 50, 50], n_folds: int = 5):
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
    
    
    # ------------ #
    # TREE model
    # ------------ #
    def tree(self, alpha: list[float, float, int] = [0, 0.001, 5], impurity: list = [0, 0.00001, 5],
             n_folds: int = 5):
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


    # ------------ #
    # Random forest
    # ------------ #
    def random_forest(self, n_estimators: list[int, int, int] = [50, 1000, 5], n_folds: int = 2):
        n_estimators = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = [{"n_estimators": n_estimators}]
        random_forest = RandomForestRegressor(warm_start=True, n_jobs=-1)
        reg_random_forest = GridSearchCV(random_forest, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_random_forest.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__random_forest_fit = reg_random_forest.best_estimator_
        # return step 
        return self.__random_forest_fit


    # ------------ #
    # Xgboost
    # ------------ #
    def xgboost(self, n_estimators: list[int, int, int] = [10, 100, 5], n_folds: int = 2):
        n_est = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = {"n_estimators": n_est}
        xgb_m = xgb.XGBRegressor(objective ='reg:squarederror')
        reg_xgb = GridSearchCV(xgb_m, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='r2')
        reg_xgb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__xgb_fit = reg_xgb.best_estimator_
        # return step 
        return self.__xgb_fit


    # ------------ #
    # model metrics
    # ------------ #
    def model_metrics(self, model_name: str = 'all'):
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
                string = f'### {key} model metrics:'
                display(Markdown(string))
                display(metrics)


    # ------------ #
    # auto_ml
    # ------------ #
    def auto_ml(self, n_folds: int = 5, data: str = 'train'):
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


    # ------------ #
    # save model
    # ------------ #
    def save_model(self, model_name: str = 'all'):
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



# ----------------- #
# view model metrics on data passed
# ----------------- #
def regression_metrics(model, X, y):
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