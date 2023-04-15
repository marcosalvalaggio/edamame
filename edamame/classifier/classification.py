#TODO - add SVM method

import numpy as np
import pandas as pd
from  edamame.eda.tools import dataframe_review, dummy_control
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
from IPython.display import display, Markdown
import matplotlib.pyplot as plt 
from typing import Tuple, Literal, List
from sklearn.svm import SVC
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class TrainClassifier:
    """
    This class represents a pipeline for training and handling classification models.

    Attributes: 
        X_train (pd.DataFrame): The input training data.
        y_train (pd.Series): The target training data.
        X_test (pd.DataFrame): The input test data.
        y_test (pd.Series): The target test data.
        
    Example:
        >>> from edamame.classifier import TrainClassifier
        >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        >>> logistic = classifier.logistic()
        >>> classifier.model_metrics(model_name="logisitc")
        >>> classifier.model_save(model_name="logisitc")
        >>> nb = classifier.gaussian_nb()
        >>> knn = classifier.knn()
        >>> tree = classifier.tree()
        >>> rf = classifier.random_forest()
        >>> xgb = classifier.xgboost()
        >>> svm = classifier.svm()
        >>> classifier.model_metrics()
        >>> # using AutoML
        >>> models = classifier.auto_ml()
        >>> classifier.save_model()
    """
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
        # init the model 
        self.__logistic_fit = {}
        self.__gaussian_nb_fit = {}
        self.__knn_fit = {}
        self.__tree_fit = {}
        self.__random_forest_fit = {}
        self.__xgb_fit = {}
        self.__svm_fit = {}


    def logistic(self) -> LogisticRegression:
        """
        Trains a logistic regression model using the training data and returns the fitted model.

        Returns:
            LogisticRegression: The trained logistic regression model.

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> logistic = classifier.logistic()
        """
        logistic = LogisticRegression()
        logistic.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__logistic_fit = logistic
        # return step 
        return self.__logistic_fit


    def gaussian_nb(self) -> GaussianNB:
        """
        Trains a Gaussian Naive Bayes classifier using the training data and returns the fitted model.

        Returns:
            GaussianNB: The trained Gaussian Naive Bayes classifier.
        
        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> nb = classifier.gaussian_nb()
        """
        gauss_nb = GaussianNB()
        gauss_nb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__gaussian_nb_fit = gauss_nb
        # return step 
        return self.__gaussian_nb_fit


    # ------------ #
    # KNN
    # ------------ #
    def knn(self, n_neighbors: Tuple[int, int, int] = (1, 50, 50), n_folds: int = 5) -> KNeighborsClassifier:
        """
        Train a k-Nearest Neighbors classification model using the training data, and perform a grid search to find the
        best value of 'n_neighbors' hyperparameter. 

        Args:
            n_neighbors (Tuple[int, int, int]): A tuple with three integers. The first and second integers are the range of the 
                'n_neighbors' hyperparameter that will be searched by the grid search, and the third integer is the 
                number of values to generate in the interval [n_neighbors[0], n_neighbors[1]]. Default is [1, 50, 50].
            n_folds (int): The number of cross-validation folds to use for the grid search. Default is 5.

        Returns:
            KNeighborsClassifier: The trained k-Nearest Neighbors classification model with the best 'n_neighbors' 
                hyperparameter found by the grid search. 
        
        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> knn = classifier.knn(n_neighbors=(1,50,50), n_folds=3) 
        """
        n_n = np.linspace(n_neighbors[0], n_neighbors[1], n_neighbors[2]).astype(np.int32)
        knn = KNeighborsClassifier()
        tuned_parameters = [{"n_neighbors": n_n}]
        reg_knn = GridSearchCV(knn, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='accuracy')
        reg_knn.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__knn_fit = reg_knn.best_estimator_
        # return step 
        return self.__knn_fit


    def tree(self, alpha: Tuple[float, float, int] = (0., 0.001, 5), impurity: Tuple[float, float, int] = (0., 0.00001, 5), n_folds: int = 5) -> DecisionTreeClassifier:
        """
        Trains a decision tree classifier using the training data and returns the fitted model.
    
        Args:
            alpha (Tuple[float, float, int]): A tuple containing the minimum and maximum values of ccp_alpha and the number of values to try (default: (0., 0.001, 5)).
            impurity (Tuple[float, float, int]): A tuple containing the minimum and maximum values of min_impurity_decrease and the number of values to try (default: (0., 0.00001, 5)).
            n_folds (int): The number of cross-validation folds to use for grid search (default: 5).

        Returns:
            DecisionTreeClassifier: The trained decision tree classifier model.

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> tree = classifier.tree(alpha=(0., 0.001, 5), impurity=(0., 0.00001, 5), n_folds=3) 
        """
        alphas = np.linspace(alpha[0], alpha[1], alpha[2])
        impurities = np.linspace(impurity[0], impurity[1], impurity[2])
        tuned_parameters = [{"ccp_alpha": alphas, 'min_impurity_decrease': impurities}]
        tree = DecisionTreeClassifier() 
        reg_tree = GridSearchCV(tree, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='accuracy')
        reg_tree.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__tree_fit = reg_tree.best_estimator_
        # return step 
        return self.__tree_fit


    def random_forest(self, n_estimators: Tuple[int, int, int] = (50, 1000, 5), n_folds: int = 2) -> RandomForestClassifier:
        """
        Train a Random Forest classifier using the training data and return the fitted model.

        Args:
            n_estimators (Tuple[int, int, int]): The range of the number of trees in the forest. Default is (50, 1000, 5).
            n_folds (int): The number of folds in cross-validation. Default is 2.

        Returns:
            RandomForestClassifier: The trained Random Forest classifier.

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> rf = classifier.random_forest(n_estimators=(50, 1000, 5), n_folds=2) 
        """
        n_estimators = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = [{"n_estimators": n_estimators}]
        random_forest = RandomForestClassifier(warm_start=True, n_jobs=-1)
        reg_random_forest = GridSearchCV(random_forest, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='accuracy')
        reg_random_forest.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__random_forest_fit = reg_random_forest.best_estimator_
        # return step 
        return self.__random_forest_fit


    def xgboost(self, n_estimators: Tuple[int, int, int] = (10, 100, 5), n_folds: int = 2) -> XGBClassifier:
        """
        Train an XGBoost classifier using the training data and return the fitted model.

        Args:
            n_estimators (Tuple[int, int, int]): The range of the number of boosting rounds. Default is (10, 100, 5).
            n_folds (int): The number of folds in cross-validation. Default is 2.

        Returns:
            XGBClassifier: The trained XGBoost classifier.

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> xgboost = classifier.xgboost(n_estimators=(10, 100, 5), n_folds=2) 
        """
        n_est = np.linspace(n_estimators[0], n_estimators[1], n_estimators[2]).astype(np.int16)
        tuned_parameters = {"n_estimators": n_est}
        xgb_m = XGBClassifier()
        reg_xgb = GridSearchCV(xgb_m, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='accuracy')
        reg_xgb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.__xgb_fit = reg_xgb.best_estimator_
        # return step 
        return self.__xgb_fit
    

    def svm(self, kernel: Literal["linear", "poly", "rbf", "sigmoid", "precomputed"] = "rbf", *args, **kwargs) -> SVC:
        """
        Trains an SVM classifier using the training data and returns the fitted model.

        Args:
            kernel (Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]): The kernel type to be used in the algorithm. Default is "rbf".
            *args: Variable length argument list to be passed to the `SVC` constructor.
            **kwargs: Arbitrary keyword arguments to be passed to the `SVC` constructor.
        
        Returns:
            SVC: The trained SVM classifier.
        
        Example: 
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> svm = classifier.svm(kernel="linear", C=1.0, gamma="auto")
        """
        svm_c = SVC(kernel=kernel, *args, **kwargs)
        svm_c.fit(self.X_train, self.y_train.squeeze())
        self.__svm_fit = svm_c
        return self.__svm_fit


    def model_metrics(self, model_name: Literal["all", "logistic", "guassian_nb", "knn", "tree", "random_forest", "xgboost", "svm"] = 'all') -> None:
        """
        Display classification metrics (confusion matrix and classification report) for specified or all trained models.

        Args:
            model_name (Literal["all", "logistic", "guassian_nb", "knn", "tree", "random_forest", "xgboost", "svm"]): The name of the model to display the metrics for. Defaults to 'all'.

        Returns:
            None

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> xgboost = classifier.xgboost(n_estimators=(10, 100, 5), n_folds=2) 
            >>> classifier.model_metrics(model_name="xgboost")
        """
        model_dct = {'logistic': 0, 'guassian_nb': 1, 'knn': 2, 'tree': 3, 'random_forest': 4, 'xgboost': 5, 'svm': 6}
        model_list = [self.__logistic_fit, self.__gaussian_nb_fit, self.__knn_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit, self.__svm_fit]
        if model_name == 'all':
            for key in model_dct:
                if model_list[model_dct[key]].__class__.__name__ == 'dict':
                        display(f'unable to show {key} model metrics')
                else:
                    title = f'### {key} model metrics:'
                    display(Markdown(title))
                    y_pred_train = model_list[model_dct[key]].predict(self.X_train)
                    y_pred_test = model_list[model_dct[key]].predict(self.X_test)
                    plt.figure(figsize=(10,4))
                    plt.subplot(121)
                    sns.heatmap(confusion_matrix(self.y_train, y_pred_train), annot=True, fmt="2.0f")
                    plt.title(f'{key} train')
                    plt.subplot(122)
                    sns.heatmap(confusion_matrix(self.y_test, y_pred_test), annot=True, fmt="2.0f")
                    plt.title(f'{key} test')
                    plt.show()
                    title = f'#### Train classification report'
                    display(Markdown(title))
                    print(classification_report(self.y_train, y_pred_train))
                    title = f'#### Test classification report'
                    display(Markdown(title))
                    print(classification_report(self.y_test, y_pred_test))
        else:
            if model_list[model_dct[model_name]].__class__.__name__ == 'dict':
                display(f'unable to show {model_name} model metrics')
            else:
                title = f'### {model_name} model metrics:'
                display(Markdown(title))
                y_pred_train = model_list[model_dct[model_name]].predict(self.X_train)
                y_pred_test = model_list[model_dct[model_name]].predict(self.X_test)
                plt.figure(figsize=(10,4))
                # plt.subplot(121)
                sns.heatmap(confusion_matrix(self.y_train, y_pred_train), annot=True, fmt="2.0f")
                plt.title(f'{model_name} train')
                plt.subplot(122)
                sns.heatmap(confusion_matrix(self.y_test, y_pred_test), annot=True, fmt="2.0f")
                plt.title(f'{model_name} test')
                plt.show()
                title = f'#### Train classification report'
                display(Markdown(title))
                print(classification_report(self.y_train, y_pred_train))
                title = f'#### Test classification report'
                display(Markdown(title))
                print(classification_report(self.y_test, y_pred_test))


    def auto_ml(self, n_folds: int = 5, data: Literal['train', 'test'] = 'train') -> List:
        """
        Perform automated machine learning with cross validation on a list of classification models.
        
        Args:
            n_folds (int): Number of cross-validation folds. Defaults to 5.
            data (Literal['train', 'test']): Target dataset for cross-validation. 
                Must be either 'train' or 'test'. Defaults to 'train'.
        
        Returns:
            List: List of best-fit classification models for each algorithm.

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> model_list = classifier.auto_ml()
        """
        kfold = KFold(n_splits=n_folds)
        cv_mean = []
        score = []
        std = []
        classifier = ["Logistic", "Gaussian NB", "KNN", "Tree", "Random forest", "Xgboost", "SVM"]
        try:
            model_list = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(n_neighbors=self.__knn_fit.n_neighbors),
                          DecisionTreeClassifier(ccp_alpha=self.__tree_fit.ccp_alpha, min_impurity_decrease=self.__tree_fit.min_impurity_decrease),
                          RandomForestClassifier(n_estimators=self.__random_forest_fit.n_estimators, warm_start=True, n_jobs=-1), 
                          XGBClassifier(n_estimators=self.__xgb_fit.n_estimators),
                          SVC(kernel=self.__svm_fit.kernel)]
        except:
            # find best hyperparameters
            self.logistic()
            self.gaussian_nb()
            self.knn()
            self.tree()
            self.random_forest()
            self.xgboost()
            # model list 
            model_list = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(n_neighbors=self.__knn_fit.n_neighbors),
                          DecisionTreeClassifier(ccp_alpha=self.__tree_fit.ccp_alpha, min_impurity_decrease=self.__tree_fit.min_impurity_decrease),
                          RandomForestClassifier(n_estimators = self.__random_forest_fit.n_estimators, warm_start=True, n_jobs=-1),
                          XGBClassifier(n_estimators = self.__xgb_fit.n_estimators),
                          SVC()]
        # cross validation loop 
        for model in model_list:
            if data == 'train':
                cv_result = cross_val_score(model, self.X_train, self.y_train.squeeze(), cv=kfold, scoring="accuracy")
            elif data == 'test':
                cv_result = cross_val_score(model, self.X_test, self.y_test.squeeze(), cv=kfold, scoring="accuracy")
            else:
                raise ValueError('insert valid target dataset (\'train\' or \'test\')')
            cv_mean.append(cv_result.mean())
            std.append(cv_result.std())
            score.append(cv_result)
        # dataframe for results 
        df_kfold_result = pd.DataFrame({"CV Mean": cv_mean, "Std": std}, index=classifier)
        # display step 
        string = f'### Metrics results on {data} set:'
        display(Markdown(string))
        display(df_kfold_result)
        # boxplot on R2
        box = pd.DataFrame(score, index=classifier)
        plt.figure(figsize=(10,8))
        box.T.boxplot()
        plt.show()

        return [self.__logistic_fit, self.__gaussian_nb_fit, self.__knn_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit, self.__svm_fit]


    def save_model(self, model_name: Literal["all", "logistic", "guassian_nb", "knn", "tree", "random_forest", "xgboost", "svm"] = 'all') -> None:
        """
        Saves the specified machine learning model or all models in the instance to a pickle file.

        Args:
            model_name (Literal["all", "linear", "lasso", "ridge", "tree", "random_forest", "xgboost", "svm"]): 
                The name of the model to save. Defaults to 'all'.
            
        Returns:
            None

        Example:
            >>> from edamame.classifier import TrainClassifier
            >>> classifier = TrainClassifier(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            >>> model_list = classifier.auto_ml()
            >>> classifier.save_model(model_name="all")
        """
        model_dct = {'logistic': 0, 'guassian_nb': 1, 'knn': 2, 'tree': 3, 'random_forest': 4, 'xgboost': 5, 'svm': 6}
        model_list = [self.__logistic_fit, self.__gaussian_nb_fit, self.__knn_fit, self.__tree_fit, self.__random_forest_fit, self.__xgb_fit, self.__svm_fit]
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
        