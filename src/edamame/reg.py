import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from IPython.display import display, Markdown
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, get_scorer_names
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict, cross_val_score
import pickle
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# list of metrics
#get_scorer_names()

# ----------------- #
# OHE check
# ----------------- #
def dummy_control(data):
    types = data.dtypes
    qual_col = types[types == 'object']
    if len(qual_col) != 0:
        raise TypeError('dataframe with non-numerical columns')
    else:
        pass

# ----------------- #
#  pandas dataframe check
# ----------------- #
def dataframe_review(data) -> None:
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')


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


    # ------------ #
    # linear model
    # ------------ #
    def linear(self):
        linear = LinearRegression()
        linear.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.linear = linear
        # return step 
        return linear


    # aggiungere sognificatività coef
    def linear_coef(self):
        intercept = ('intercept',self.linear.intercept_[0])
        coef = list(zip(list(self.X_train.columns), self.linear.coef_[0]))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Linear Coef']
        # display step 
        display(df_coef)
        

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
        self.lasso = reg_lasso.best_estimator_
        # return step 
        return lasso


    def lasso_coef(self):
        intercept = ('intercept',self.lasso.intercept_[0])
        coef = list(zip(list(self.X_train.columns), self.lasso.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Lasso Coef']
        # display step 
        display(df_coef)
            
        
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
        self.ridge = reg_ridge.best_estimator_
        # return step 
        return ridge
    
    
    def ridge_coef(self):
        intercept = ('intercept',self.ridge.intercept_[0])
        coef = list(zip(list(self.X_train.columns), self.ridge.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Ridge Coef']
        # display step 
        display(df_coef)
        
    
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
        self.tree = reg_tree.best_estimator_
        # return step 
        return tree


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
        self.random_forest = reg_random_forest.best_estimator_
        # return step 
        return random_forest


    def random_forest_fi(self, figsize: tuple[float, float] = (12,10)):
        importances = self.random_forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.random_forest.estimators_], axis=0)
        feature_names = self.random_forest.feature_names_in_
        forest_importances = pd.Series(importances, index=feature_names)
        plt.figure(figsize=figsize)
        forest_importances.plot.bar(yerr=std)
        plt.title("Feature importances using mean decrease in impurity")
        plt.ylabel("Mean decrease in impurity")
        plt.show()
    

    # ------------ #
    # model metrics
    # ------------ #
    def model_metrics(self, model_name: str = 'all'):
        model_dct = {'linear': 0, 'lasso': 1, 'ridge': 2, 'tree': 3, 'random forest': 4}
        model_list = [self.linear, self.lasso, self.ridge, self.tree, self.random_forest]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'method':
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
            if model_list[model_dct[model_name]].__class__.__name__ == 'method':
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
    def auto_ml(self, n_folds: int = 5, data: str = 'test'):
        kfold = KFold(n_splits=n_folds)
        cv_mean = []
        score = []
        std = []
        regressor = ["Linear", "Lasso", "Ridge", "Tree", "Random Forest"]
        try:
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.tree.ccp_alpha, min_impurity_decrease=self.tree.min_impurity_decrease),
                          RandomForestRegressor(n_estimators = self.random_forest.n_estimators, warm_start=True, n_jobs=-1)]
        except:
            # find best hyperparameters
            self.linear()
            self.lasso()
            self.ridge()
            self.tree()
            self.random_forest()
            # model list 
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.tree.ccp_alpha, min_impurity_decrease=self.tree.min_impurity_decrease),
                          RandomForestRegressor(n_estimators = self.random_forest.n_estimators, warm_start=True, n_jobs=-1)]
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
        df_kfold_result = pd.DataFrame({"CV Mean": cv_mean, "Std": std},index=regressor)
        # display step 
        string = f'### Metrics results on {data} set:'
        display(Markdown(string))
        display(df_kfold_result)
        # boxplot on R2
        box = pd.DataFrame(score, index=regressor)
        box.T.boxplot()
        plt.show()


    # ------------ #
    # save model
    # ------------ #
    def save_model(self, model_name: str = 'all'):
        model_dct = {'linear': 0, 'lasso': 1, 'ridge': 2, 'tree': 3, 'random forest': 4}
        model_list = [self.linear, self.lasso, self.ridge, self.tree, self.random_forest]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'method':
                        display(f'unable to save {key} model')
                else:
                    filename = f'{key}.pkl'
                    with open(filename, 'wb') as file:
                        pickle.dump(model_list[model_dct[key]], file)
                        display(f'{filename} saved')
        else:
            if model_list[model_dct[model_name]].__class__.__name__ == 'method':
                display(f'unable to save {model_name} model')
            else:
                filename = f'{model_name}.pkl'
                with open(filename, 'wb') as file:
                    pickle.dump(model_list[model_dct[model_name]], file)



# ----------------- #
# load model 
# ----------------- #
def load_model(path: str):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model 

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # OHE formatting
    X_train_t = pd.get_dummies(data=X_train, drop_first=False)
    X_test_t = pd.get_dummies(data=X_test, drop_first=False)
    # instance
    regressor = TrainRegressor(X_train_t, np.log(y_train), X_test_t, np.log(y_test))
    regressor.linear()
    regressor.linear_coef()
    regressor.auto_ml()
    regressor.model_metrics()
    regressor.save_model()