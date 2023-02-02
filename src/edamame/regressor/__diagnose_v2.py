import pandas as pd 
from IPython.display import display
import numpy as np 
import matplotlib.pyplot as plt 
from yellowbrick.regressor import prediction_error


class Diagnose():

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model 
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def check_model(self, model_name: str) -> None:
        if self.model.__class__.__name__ != model_name:
            raise TypeError('The model passed isn\'t a linear model')


    def coef(self): 
        intercept = ('intercept', self.model.intercept_)
        coef = list(zip(list(self.model.feature_names_in_), self.model.coef_))
        coef = [intercept, *coef]
        df_coef = pd.DataFrame(coef)
        df_coef.columns = ['Columns', 'Coef']
        # display step 
        display(df_coef)
    

    def model_predict_error(self):
        visualizer = prediction_error(self.model, self.X_train, self.y_train, self.X_test, self.y_test)


class LinearDiagnose(Diagnose):

    def __init__(self, model, X_train, y_train, X_test, y_test):
        super().__init__(model, X_train, y_train, X_test, y_test)


    def check_model(self, model_name: str = "LinearRegression") -> None:
        return super().check_model(model_name)


class LassoDiagnose(Diagnose):

    def __init__(self, model, X_train, y_train, X_test, y_test):
        super().__init__(model, X_train, y_train, X_test, y_test)


    def check_model(self, model_name: str = "Lasso") -> None:
        return super().check_model(model_name)


class RidgeDiagnose(Diagnose):

    def __init__(self, model, X_train, y_train, X_test, y_test):
        super().__init__(model, X_train, y_train, X_test, y_test)


    def check_model(self, model_name: str = "Ridge") -> None:
        return super().check_model(model_name)
    


