

# Edamame

[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://edamame-doc.readthedocs.io/en/latest/index.html)

Edamame is inspired by packages such as [pandas-profiling](https://github.com/ydataai/pandas-profiling), [pycaret](https://github.com/pycaret/pycaret>), and [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick>). The goal of Edamame is to provide user-friendly functions for conducting exploratory data analysis (EDA) on datasets, as well as for training and analyzing batteries of models for regression or classification problems.

To install the package,

```console
pip install edamame
```

the edamame package works correctly inside a jupyter-notebook. You can find the documentation of the package on the [edamame-documentation](https://edamame-doc.readthedocs.io/en/latest/index.html) page.

<hr>


## Edamame functionalities


The package consists of three modules: eda, which performs exploratory data analysis; and regressor and classifier, which handle the training of machine learning models for regression and classification, respectively. To see examples of the uses of the edamame package, you can check out the [edamame-notebook](https://github.com/marcosalvalaggio/edamame-notebooks) repository.

<hr>

## Eda module


``import edamame.eda as eda``

The **eda** module provides a wide range of functions for performing exploratory data analysis (EDA) on datasets. With this module you can easily explore and manipulate your data, conduct descriptive statistics, correlation analysis, and prepare your data for machine learning. The "eda" module offers the following functionalities:

* Data Exploration and Manipulation functions:

   - **dimensions**: The function displays the number of rows and columns of a pandas dataframe passed. 
   - **identify_types**: Identify the data types of each column.
   - **view_cardinality**: View the number of unique values in each categorical column.
   - **modify_cardinality**: Modify the number of unique values in a column.
   - **missing**: Check if any missing data is present in the dataset.
   - **handling_missing**: Replace or remove missing values in the dataset.
   - **drop_columns**: Remove specific columns from the dataset.
   - **num_to_categorical**: The function returns a dataframe with the columns transformed into an "object".
   - **interaction**: The function display an interactive plot for analysing relationships between numerical columns with a scatterplot.
   - **inspection**: The function displays an interactive plot for analysing the distribution of a variable based on the distinct cardinalities of the target variable.
   - **split_and_scaling**: The function returns two pandas dataframes: the regressor matrix X contains all the predictors for the model, the series y contains the values of the response variable.

* Descriptive Statistics functions:

   - **describe_distribution**: The function display the result of the describe() method applied to a pandas dataframe, divided by numerical and object columns.
   - **plot_categorical**: The function returns a sequence of tables and plots for the categorical variables.
   - **plot_numerical**:  The function returns a sequence of tables and plots for the numerical variables.
   - **num_variable_study**: he function displays the following transformations of the variable col passed: log(x), sqrt(x), x^2, Box-cox, 1/x.

* Correlation Analysis functions:

   - **correlation_pearson**: The function performs the Pearson's correlation between the columns pairs. 
   - **correlation_categorical**: The function performs the Chi-Square Test of Independence between categorical variables of the dataset. 
   - **correlation_phik**: Calculate the Phik correlation coefficient between all pairs of columns ([Paper link](https://arxiv.org/pdf/1811.11440.pdf)).

* Useful function:

   - **load_model**: The function load the model saved in the pickle format.
   - **setup**: The function returns the following elements: X_train, y_train, X_test, y_test.
   - **scaling**: The function returns the normalised/standardized matrix.

<hr>

## Regressor module

```python
from edamame.regressor import TrainRegressor, regression_metrics
```

The TrainRegressor class is designed to be used as a pipeline for training and handling regression models.

The class provides several methods for fitting different regression models, computing model metrics, saving and loading models, and using AutoML to select the best model based on performance metrics. These methods include:

* **linear**: Fits a linear regression model to the training data.
* **lasso**: Fits a Lasso regression model to the training data.
* **ridge**: Fits a Ridge regression model to the training data.
* **tree**: Fits a decision tree regression model to the training data.
* **random_forest**: Fits a random forest regression model to the training data.
* **xgboost**: Fits an XGBoost regression model to the training data.
* **auto_ml**: Uses AutoML to select the best model based on performance metrics.
* **model_metrics**: Computes and prints the performance metrics for each trained model.
* **save_model**: Saves the trained model to a file.

After saving a model with the **save_model** method, we can upload the model using the **load_model** function of the eda module and evaluate its performance on new data using the **regression_metrics** function.

```python
from edamame.regressor import RegressorDiagnose
```

The RegressorDiagnose class is designed to diagnose regression models and analyze their performance.
The class provides several methods for diagnosing and analyzing the performance of regression models. These methods include:

* **coefficients**: Computes and prints the coefficients of the regression model.
* **random_forest_fi**: Displays the feature importance plot for the random forest regression model. 
* **random_forest_fi**: Displays the feature importance plot for the xgboost regression model. 
* **prediction_error**: Computes and prints the prediction error of the regression model on the test data.
* **residual_plot**: creates and displays a residual plot for the regression model.
* **qqplot**: creates and displays a QQ plot for the regression model.

<hr>

## Classifier module


```python
from edamame.classifier import TrainClassifier
```

The TrainClassifier class is designed to be used as a pipeline for training and handling clasification models.

The class provides several methods for fitting different regression models, computing model metrics, saving and loading models, and using AutoML to select the best model based on performance metrics. These methods include:

* **logistic**: Fits a logistic model to the training data.
* **gaussian_nb**: Fits a Gaussina Naive Bayes model to the training data.
* **knn**: Fits a k-Nearest Neighbors classification model to the training data.
* **tree**: Fits a decision tree classification model to the training data.
* **random_forest**: Fits a random forest classification model to the training data.
* **xgboost**: Fits an XGBoost classification model to the training data.
* **auto_ml**: Uses AutoML to select the best model based on performance metrics.
* **model_metrics**: Computes and prints the performance metrics for each trained model.
* **save_model**: Saves the trained model to a file.

<hr>

## Todos


* Add support for SVM to the classifier module.
* Add the ClassifierDiagnose class to the classifier module.
* Add the notebook for EDA in a classification problem to the edamame-notebook repository.
* Add the notebook for training/diagnosing classification models to the edamame-notebook repository.
* **Add link to docs url**.


