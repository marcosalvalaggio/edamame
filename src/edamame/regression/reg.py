import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from IPython.display import display
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    
    def __init__(self, X, y, n_folds: int = 5, alpha_lasso: list[float, float, int]=[0.0001,10,50], 
                 alpha_ridge: list[float, float, int]=[0.0001,10,50]):
        self.X = X
        self.y = y
        # check dataframe 
        dataframe_review(self.X)
        # check columns type
        dummy_control(self.X)
        self.n_folds = n_folds
        # lasso hyperparameter 
        self.alpha_lasso = np.linspace(alpha_lasso[0], alpha_lasso[1], alpha_lasso[2])
        # ridge hyperparameter
        self.alpha_ridge = np.linspace(alpha_ridge[0], alpha_ridge[1], alpha_ridge[2])
    
    # ------------ #
    # linear model
    # ------------ #
    def linear(self):
        linear = LinearRegression()
        linear.fit(self.X, self.y)
        # save the model in the instance attributes
        self.linear = linear
        # return step 
        return linear

    # aggiungere sognificatività coef
    def linear_coef(self):
        intercept = ('intercept',self.linear.intercept_[0])
        coef = list(zip(list(self.X.columns), self.linear.coef_[0]))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Linear Coef']
        # display step 
        display(df_coef)
        
    def linear_metrics(self):
        # R2
        y_pred_train = self.linear.predict(self.X)
        r2 = r2_score(self.y, y_pred_train)
        # MSE
        mse = mean_squared_error(self.y, y_pred_train)
        # MAE
        mae = mean_absolute_error(self.y, y_pred_train)
        # display step 
        index_label = ['R2', 'MSE', 'MAE']
        metrics = pd.DataFrame([r2, mse, mae], index = index_label)
        metrics.columns = ['Linear']
        display(metrics)


    # ------------ #
    # Lasso model
    # ------------ #
    def lasso(self):
        lasso = Lasso()
        tuned_parameters = [{"alpha": self.alpha_lasso}]
        reg_lasso = GridSearchCV(lasso, tuned_parameters, cv=self.n_folds, refit=True, verbose = 0, scoring='r2')
        reg_lasso.fit(self.X, self.y)
        # running the optimized model 
        lasso = Lasso(alpha = reg_lasso.best_params_['alpha'])
        lasso.fit(self.X, self.y)
        # save the model in the instance attributes
        self.lasso = lasso
        # return step 
        return lasso

    def lasso_coef(self):
        intercept = ('intercept',self.lasso.intercept_[0])
        coef = list(zip(list(self.X.columns), self.lasso.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Lasso Coef']
        # display step 
        display(df_coef)
    
    def lasso_metrics(self):
        # R2
        y_pred_train = self.lasso.predict(self.X)
        r2 = r2_score(self.y, y_pred_train)
        # MSE
        mse = mean_squared_error(self.y, y_pred_train)
        # MAE
        mae = mean_absolute_error(self.y, y_pred_train)
        # display step 
        index_label = ['R2', 'MSE', 'MAE']
        metrics = pd.DataFrame([r2, mse, mae], index = index_label)
        metrics.columns = [f'Lasso with alpha: {self.lasso.alpha:.4f}']
        display(metrics)        
        
    # ------------ #
    # Ridge model
    # ------------ #
    def ridge(self):
        ridge = Ridge()
        tuned_parameters = [{"alpha": self.alpha_ridge}]
        reg_ridge = GridSearchCV(ridge, tuned_parameters, cv=self.n_folds, refit=True, verbose = 0, scoring='r2')
        reg_ridge.fit(self.X, self.y)
        # running the optimized model 
        ridge = Ridge(alpha = reg_ridge.best_params_['alpha'])
        ridge.fit(self.X, self.y)
        # save the model in the instance attributes
        self.ridge = ridge
        # return step 
        return ridge
    
    
    def ridge_coef(self):
        intercept = ('intercept',self.ridge.intercept_[0])
        coef = list(zip(list(self.X.columns), self.ridge.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Ridge Coef']
        # display step 
        display(df_coef)
        
    
    def ridge_metrics(self):
        # R2
        y_pred_train = self.ridge.predict(self.X)
        r2 = r2_score(self.y, y_pred_train)
        # MSE
        mse = mean_squared_error(self.y, y_pred_train)
        # MAE
        mae = mean_absolute_error(self.y, y_pred_train)
        # display step 
        index_label = ['R2', 'MSE', 'MAE']
        metrics = pd.DataFrame([r2, mse, mae], index = index_label)
        metrics.columns = [f'Ridge with alpha: {self.ridge.alpha:.4f}']
        display(metrics) 

    # ------------ #
    # auto_ml
    # ------------ #
    def auto_ml(self):
        kfold = KFold(n_splits=self.n_folds)
        cv_mean = []
        score = []
        std = []
        regressor = ["Linear", "Lasso", "Ridge"]
        try:
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha)]
        except:
            # find best hyperparameters 
            self.lasso()
            self.ridge()
            # model list 
            model_list = [LinearRegression(), Lasso(alpha = self.lasso.alpha),
                          Ridge(alpha = self.ridge.alpha)]
        # cross validation loop 
        for model in model_list:
            cv_result = cross_val_score(model, self.X, self.y, cv=kfold, scoring="r2")
            cv_mean.append(cv_result.mean())
            std.append(cv_result.std())
            score.append(cv_result)
        # dataframe for results 
        df_kfold_result = pd.DataFrame({"CV Mean": cv_mean, "Std": std},index=regressor)
        # display step 
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