import numpy as np
import pandas as pd
from .tools import dataframe_review, dummy_control
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
from IPython.display import display, Markdown
import matplotlib.pyplot as plt 
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class TrainClassifier:

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
        self.logistic_fit = {}
        self.gaussian_nb_fit = {}
        self.knn_fit = {}

    # ------------ #
    # logistic model
    # ------------ #
    def logistic(self):
        logistic = LogisticRegression()
        logistic.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.logistic_fit = logistic
        # return step 
        return self.logistic_fit


    # ------------ #
    # Naive gaussian Bayes
    # ------------ #
    def gaussian_nb(self):
        gauss_nb = GaussianNB()
        gauss_nb.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.gaussian_nb_fit = gauss_nb
        # return step 
        return self.gaussian_nb_fit


    # ------------ #
    # KNN
    # ------------ #
    def knn(self, n_neighbors: int = [1, 50, 50], n_folds: int = 5):
        n_n = np.linspace(n_neighbors[0], n_neighbors[1], n_neighbors[2]).astype(np.int32)
        knn = KNeighborsClassifier()
        tuned_parameters = [{"n_neighbors": n_n}]
        reg_knn = GridSearchCV(knn, tuned_parameters, cv=n_folds, refit=True, verbose=0, scoring='accuracy')
        reg_knn.fit(self.X_train, self.y_train.squeeze())
        # save the model in the instance attributes
        self.knn_fit = reg_knn.best_estimator_
        # return step 
        return self.knn_fit

    
    # ------------ #
    # model metrics
    # ------------ #
    def model_metrics(self, model_name: str = 'all'):
        model_dct = {'logistic': 0, 'guassian_nb': 1, 'knn': 2}
        model_list = [self.logistic_fit, self.gaussian_nb_fit, self.knn_fit]
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
                plt.subplot(121)
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


    # ------------ #
    # auto_ml
    # ------------ #
    def auto_ml(self, n_folds: int = 5, data: str = 'test'):
        kfold = KFold(n_splits=n_folds)
        cv_mean = []
        score = []
        std = []
        classifier = ["Logistic", "Gaussian NB", "KNN"]
        try:
            model_list = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(n_neighbors=self.knn_fit.n_neighbors)]
        except:
            # find best hyperparameters
            self.logistic()
            self.gaussian_nb()
            self.knn()
            # model list 
            model_list = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(n_neighbors=self.knn_fit.n_neighbors)]
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
        box.T.boxplot()
        plt.show()


    # ------------ #
    # save model
    # ------------ #
    def save_model(self, model_name: str = 'all'):
        model_dct = {'logistic': 0, 'guassian_nb': 1, 'knn': 2}
        model_list = [self.logistic_fit, self.gaussian_nb_fit, self.knn_fit]
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
        