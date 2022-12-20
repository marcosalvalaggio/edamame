import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from IPython.display import display, Markdown
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, get_scorer_names
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict, cross_val_score
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# list of metrics
#get_scorer_names()

def dummy_control(data):
    types = data.dtypes
    qual_col = types[types == 'object']
    if len(qual_col) != 0:
        raise TypeError('dataframe with non-numerical columns')
    else:
        pass


def dataframe_review(data) -> None:
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')


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
        linear.fit(self.X_train, self.y_train)
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
        

    def linear_metrics(self):
        y_pred_train = self.linear.predict(self.X_train)
        y_pred_test = self.linear.predict(self.X_test)
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
        metrics.columns = ['Train','Test']
        string = '### Linear model metrics:'
        display(Markdown(string))
        display(metrics)


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
        reg_lasso.fit(self.X_train, self.y_train)
        # running the optimized model 
        lasso = Lasso(alpha = reg_lasso.best_params_['alpha'])
        lasso.fit(self.X_train, self.y_train)
        # save the model in the instance attributes
        self.lasso = lasso
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
    

    def lasso_metrics(self):
        # R2
        y_pred_train = self.lasso.predict(self.X_train)
        y_pred_test = self.linear.predict(self.X_test)
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
        string = '### Lasso metrics:'
        display(Markdown(string))
        metrics.columns = [f'Train with alpha: {self.lasso.alpha:.4f}', 'Test']
        display(metrics)        
        

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
        reg_ridge.fit(self.X_train, self.y_train)
        # running the optimized model 
        ridge = Ridge(alpha = reg_ridge.best_params_['alpha'])
        ridge.fit(self.X_train, self.y_train)
        # save the model in the instance attributes
        self.ridge = ridge
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
        
    
    def ridge_metrics(self):
        y_pred_train = self.ridge.predict(self.X_train)
        y_pred_test = self.linear.predict(self.X_test)
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
        metrics.columns = [f'Train with alpha: {self.ridge.alpha:.4f}', 'Test']
        string = '### Ridge metrics:'
        display(Markdown(string))
        display(metrics) 


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
        reg_tree.fit(self.X_train, self.y_train)
        # running the optimized model 
        alpha = reg_tree.best_params_['ccp_alpha']
        min_impurity= reg_tree.best_params_['min_impurity_decrease']
        tree = DecisionTreeRegressor(ccp_alpha=alpha, min_impurity_decrease=min_impurity)
        tree.fit(self.X_train, self.y_train)
        # save the model in the instance attributes
        self.tree = tree
        # return step 
        return tree


    def tree_metrics(self):
        y_pred_train = self.tree.predict(self.X_train)
        y_pred_test = self.tree.predict(self.X_test)
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
        string = '### Tree metrics:'
        display(Markdown(string))
        display(metrics) 


    # ------------ #
    # auto_ml
    # ------------ #
    def auto_ml(self, n_folds: int = 5):
        kfold = KFold(n_splits=n_folds)
        cv_mean = []
        score = []
        std = []
        regressor = ["Linear", "Lasso", "Ridge", "Tree"]
        try:
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.tree.ccp_alpha, min_impurity_decrease=self.tree.min_impurity_decrease)]
        except:
            # find best hyperparameters 
            self.lasso()
            self.ridge()
            self.tree()
            # model list 
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha),
                          DecisionTreeRegressor(ccp_alpha=self.tree.ccp_alpha, min_impurity_decrease=self.tree.min_impurity_decrease)]
        # cross validation loop 
        for model in model_list:
            cv_result = cross_val_score(model, self.X_test, self.y_test, cv=kfold, scoring="r2")
            cv_mean.append(cv_result.mean())
            std.append(cv_result.std())
            score.append(cv_result)
        # dataframe for results 
        df_kfold_result = pd.DataFrame({"CV Mean": cv_mean, "Std": std},index=regressor)
        # display step 
        string = '### Metrics results:'
        display(Markdown(string))
        display(df_kfold_result)
        # boxplot on R2
        box = pd.DataFrame(score, index=regressor)
        box.T.boxplot()
        plt.show()



if __name__ == '__main__':

    X = pd.read_csv('/Users/marcosalvalaggio/code/python/ds/data/melb_data/X.csv', sep = ';')
    y = pd.read_csv('/Users/marcosalvalaggio/code/python/ds/data/melb_data/y.csv', sep = ';')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # OHE formatting
    X_train_t = pd.get_dummies(data=X_train, drop_first=False)
    # instance
    model = TrainRegressor(X_train_t, np.log(y_train))
    model.linear()
    model.linear_coef()
    model.linear_metrics()