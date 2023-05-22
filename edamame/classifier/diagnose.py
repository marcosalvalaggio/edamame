import edamame.eda as eda
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Union
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import random
from typing import Tuple
import xgboost as xgb
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc


def check_random_forest(model: RandomForestClassifier) -> None: 
    """
    The function checks if the model passed is a random forest regression.

    Args:
        model (RandomForestClassifier): The input model to be checked.

    Raises:
            TypeError: If the input model is not a random forest regression model.
    
    Returns:
        None
    """
    if model.__class__.__name__ != 'RandomForestClassifier':
        raise TypeError('The model passed isn\'t a ridge model')
    

def check_xgboost(model: XGBClassifier) -> None: 
    """
    The function checks if the model passed is a xgboost regression.

    Args:
        model (xgb.XGBRegressor): The input model to be checked.

    Raises:
        TypeError: If the input model is not an XGBoost regression model.

    Returns:
        None
    """
    if model.__class__.__name__ != 'XGBClassifier':
        raise TypeError('The model passed isn\'t an xgboost')
    

class ClassifierDiagnose:
    """
    A class for diagnosing classification models.

    Attributes:
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.

    Examples:
        >>> from edamame.classifier import TrainClassifier
        >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        >>> nb = classifier.gaussian_nb()
        >>> classifiers_diagnose = ClassifierDiagnose(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        >>> classifiers_diagnose.class_prediction_error(model=nb)
    """
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def class_prediction_error(self, model: Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC], train_data: bool = True) -> None:
        """
        This plot method shows the support (number of training samples) for each class in the fitted classification model as a stacked bar chart. Each bar is segmented to show the proportion of predictions (including false negatives and false positives, like a Confusion Matrix) for each class

        Args:
            model (Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC]): Classification model.
            train_data (bool): Defines if you want to plot the stacked barplot on train or test data (train by default).
        
        Returns: 
            None
        """
        def stacked_barplot(matrix: np.array, num_of_class: int) -> None:
            cmap = plt.get_cmap('rainbow')
            plt.figure(figsize=(8,6))
            cm_bottom = np.zeros((num_of_class,))
            for i in range(len(matrix)):
                value = random.uniform(0, 1)
                clr = cmap(value)
                if i == 0:
                    plt.bar(range(num_of_class), matrix[i], label=f'Class {i}', color=clr)
                else: 
                    cm_bottom += matrix[i-1]
                    plt.bar(range(num_of_class), matrix[i], bottom=cm_bottom, label=f'Class {i}', color=clr)
            plt.xticks(range(num_of_class), [str(i) for i in range(num_of_class)])
            plt.ylabel('Count')
            plt.title('Predicted vs True Class')
            plt.legend()
            plt.show()

        if train_data:
            y_pred = model.predict(self.X_train)
            y_true = self.y_train.squeeze().to_numpy() 
            num_classes = y_true.max() + 1
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            stacked_barplot(matrix=cm, num_of_class=num_classes)
        else:
            y_pred = model.predict(self.X_test)
            y_true = self.y_test.squeeze().to_numpy() 
            num_classes = y_true.max() + 1
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            stacked_barplot(matrix=cm, num_of_class=num_classes)
    

    def random_forest_fi(self, model: RandomForestClassifier, figsize: Tuple[float, float] = (12,10)) -> None:
        """
        The method displays the feature importance plot of the random forest model. 

        Args:
            model (RandomForestClassifier): The input random forest model.

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

    
    def xgboost_fi(self, model: XGBClassifier, figsize: tuple[float, float] = (14,12)) -> None:
        """
        The method displays the feature importance plot.

        Args:
            model (XGBClassifier): The input xgboost model.

        Returns:
            None
        """
        check_xgboost(model)
        xgb.plot_importance(model)
        plt.rcParams['figure.figsize'] = [figsize[0], figsize[1]]
        plt.show()

    
    def plot_roc_auc(self, model: Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC], train_data: bool = True) -> None: 
        """
        Method for plotting the ROC curve and calculating the AUC values for a given model.

        Args: 
            model (Union[LogisticRegression, GaussianNB, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, SVC]): Classification model.
            train_data (bool): Defines if you want to plot the stacked barplot on train or test data (train by default).
        
        Returns: 
            None
        """
        # One VS All strategy
        def _OVR_roc_curve(x, y, target_data):
            y_ohe = eda.ohe(y.iloc[:,0])
            y_score = model.predict_proba(x)
            n_classes = y_ohe.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_ohe[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Plot ROC curve for each class
            plt.figure()
            colors = ['blue', 'red', 'green']
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
            # Plot the random chance line
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            # Set plot properties
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'One-vs-Rest multiclass ROC for {target_data}')
            plt.legend(loc="lower right")
            plt.show()
        # check binary problem: 
        if len(set(self.y_train.iloc[:,0])) > 2:
            if train_data:
                _OVR_roc_curve(self.X_train, self.y_train, target_data="Training data")
            else:
                _OVR_roc_curve(self.X_test, self.y_test, target_data="Test data")
        else:
            if train_data:
                roc = nb_roc = RocCurveDisplay.from_estimator(model, self.X_train, self.y_train)
                plt.show()
            else: 
                roc = nb_roc = RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)
                plt.show()
