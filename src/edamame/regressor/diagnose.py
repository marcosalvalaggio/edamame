import pandas as pd 
from IPython.display import display


# ------------------- #
#     LINEAR MODEL 
# ------------------- #
def check_linear(model):
    if model.__class__.__name__ != 'LinearRegression':
        raise TypeError('The model passed isn\'t a linear model')
        

def linear_coef(model):
    check_linear(model)
    intercept = ('intercept',model.intercept_[0])
    coef = list(zip(list(model.feature_names_in_), model.coef_[0]))
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
    intercept = ('intercept',model.intercept_[0])
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
    intercept = ('intercept',model.intercept_[0])
    coef = list(zip(list(model.feature_names_in_), model.coef_))
    coef = [intercept, *coef]
    df_coef = pd.DataFrame(coef)
    df_coef.columns = ['Columns', 'Ridge Coef']
    # display step 
    display(df_coef)