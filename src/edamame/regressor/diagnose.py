import pandas as pd 
from IPython.display import display
import numpy as np 
import xgboost as xgb
import matplotlib.pyplot as plt 



# ------------------- #
#     LINEAR MODEL 
# ------------------- #
def check_linear(model):
    if model.__class__.__name__ != 'LinearRegression':
        raise TypeError('The model passed isn\'t a linear model')
        

def linear_coef(model):
    check_linear(model)
    intercept = ('intercept',model.intercept_)
    coef = list(zip(list(model.feature_names_in_), model.coef_))
    coef = [intercept, *coef]
    df_coef = pd.DataFrame(coef)
    df_coef.columns = ['Columns', 'Linear Coef']
    # display step 
    display(df_coef)


# ------------------- #
#     LASSO MODEL 
# ------------------- #
def check_lasso(model): 
    if model.__class__.__name__ != 'Lasso':
        raise TypeError('The model passed isn\'t a lasso model')

def lasso_coef(model):
    check_lasso(model)
    intercept = ('intercept',model.intercept_)
    coef = list(zip(list(model.feature_names_in_), model.coef_))
    coef = [intercept, *coef]
    df_coef = pd.DataFrame(coef)
    df_coef.columns = ['Columns', 'Lasso Coef']
    # display step 
    display(df_coef)



# ------------------- #
#     RIDGE MODEL 
# ------------------- #
def check_ridge(model): 
    if model.__class__.__name__ != 'Ridge':
        raise TypeError('The model passed isn\'t a ridge model')


def ridge_coef(model):
    check_ridge(model)
    intercept = ('intercept',model.intercept_)
    coef = list(zip(list(model.feature_names_in_), model.coef_))
    coef = [intercept, *coef]
    df_coef = pd.DataFrame(coef)
    df_coef.columns = ['Columns', 'Ridge Coef']
    # display step 
    display(df_coef)


# ------------------- #
#     RANDOM FOREST 
# ------------------- #
def check_random_forest(model): 
    if model.__class__.__name__ != 'RandomForestRegressor':
        raise TypeError('The model passed isn\'t a ridge model')


def random_forest_fi(model, figsize: tuple[float, float] = (12,10)):
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


# ------------------- #
#     XGBOOST 
# ------------------- #
def check_xgboost(model): 
    if model.__class__.__name__ != 'XGBRegressor':
        raise TypeError('The model passed isn\'t an xgboost')


def xgboost_fi(model, figsize: tuple[float, float] = (12,10)):
    check_xgboost(model)
    xgb.plot_importance(model)
    plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
    plt.show()