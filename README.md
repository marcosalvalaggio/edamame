

- [Installation](#installation)
- [Why Edamame?](#why-edamame)
  - [Exploratory data analysis functions](#exploratory-data-analysis-functions)
    - [Dimensions](#dimensions)
    - [Describe distribution](#describe-distribution)
    - [Identify columns types](#identify-columns-types)
    - [Convert numerical columns to categorical](#convert-numerical-columns-to-categorical)
    - [Missing data](#missing-data)
    - [Handling Missing values](#handling-missing-values)
    - [Drop columns](#drop-columns)
    - [Plot categorical variables](#plot-categorical-variables)
    - [Plot numerical variables](#plot-numerical-variables)
    - [View cardinalities of variables](#view-cardinalities-of-variables)
    - [Modify the cardinalities of a variable](#modify-the-cardinalities-of-a-variable)
    - [Distribution study of a numerical variable](#distribution-study-of-a-numerical-variable)
    - [Pearson's correlation matrix](#pearsons-correlation-matrix)
    - [Correlation matrix for categorical columns](#correlation-matrix-for-categorical-columns)
    - [Phik Correlation matrix](#phik-correlation-matrix)
    - [Interaction](#interaction)
    - [Inspection](#inspection)
    - [Split and scaling](#split-and-scaling)
  - [TODO](#todo)




# Installation 

To install the package,

```
pip install edamame
```

the edamame package works correctly inside a **.ipynb** file. 

```python
import edamame as eda
```
# Why Edamame?

Edamame is born under the inspiration of the [pandas-profiling](https://github.com/ydataai/pandas-profiling) and [pycaret](https://github.com/pycaret/pycaret) packages. The scope of edamame is to build friendly and helpful functions for handling the exploratory data analysis (EDA) step in a dataset studied and then train and analyse a model's battery for regression or classification problems. 

## Exploratory data analysis functions

You can find an example of the EDA that uses the edamame package in the [edamame-notebooks](https://github.com/marcosalvalaggio/edamame-notebooks) repository.

### Dimensions

a prettier version of the **.shape** method

```python
eda.dimensions(data)
```
Parameters: 

* **data**: A pandas dataframe

The function displays the number of rows and columns of a pandas dataframe passed. 

### Describe distribution


```python
eda.describe_distribution(data)
```

Parameters: 

* **data**: A pandas dataframe.

Passing a dataframe the function display the result of the **.describe()** method applied to a pandas dataframe, divided by quantitative/numerical and categorical/object columns.


### Identify columns types


```python
eda.identify_types(data)
```
Parameters: 

* **data**: A pandas dataframe.
  
Passing a dataframe the function display the result of the **.dtypes** method and returns a list with the name of the quantitative/numerical columns and a list with the name of the columns identified as "object" by pandas. 


### Convert numerical columns to categorical

```python
eda.num_to_categorical(data, col: list[str])
```

Parameters: 

* **data**: A pandas dataframe.
* **col**: A list of strings containing the names of columns to convert. 

Passing a dataframe and a list with columns name, the function returns a dataframe with the columns transformed into an "object". 


### Missing data

```python
eda.missing(data)
```

Parameters: 

* **data**: A pandas dataframe.

The function display the following elements:

* A table with the percentage of **NA** record for every column.
* A table with the percentage of **0** as a record for every column.
* A table with the percentage of duplicate rows.
* A list of lists that contains the name of the numerical columns with **NA**, the name of the categorical columns with **NA** and the name of the columns with 0 as a record. 

### Handling Missing values

```python
eda.handling_missing(data, col: list[str], missing_val = np.nan, method: list[str] = [])
```

Parameters: 

* **data**: A pandas dataframe.
* **col**: A list of the names of the dataframe columns to handle.
* **missing_val**: The value that represents the **NA** in the columns passed. By default is equal to **np.nan**.
* **method**: A list of the names of the methods (mean, median, most_frequent, drop) applied to the columns passed. By default, if nothing was indicated, the function applied the **most_frequent** method to all the columns passed. Indicating fewer methods than the names of the columns leads to an autocompletion with the **most_frequent** method.

The function returns a pandas dataframe with the columns selected modified to handle the nan values. 

### Drop columns 

```python 
eda.drop_columns(data, col: list[str]):
```

Parameters:

* **data**: A pandas dataframe.
* **col**: A list of strings containing the names of columns to drop. 

The function returns a pandas dataframe with the columns selected dropped.


### Plot categorical variables

```python 
eda.plot_categorical(data, col: list[str])
```
Parameters:

* **data**: A pandas dataframe
* **col**: A list of string containing the names of columns to plot
  
The function returns a sequence of tables and plots. For every variables the **plot_categorical** produce an **info** table that contains the information about: 

* The number of not nan rows. 
* The number of unique values. 
* The name of the value with the major frequency.
* The frequence of the top unique value. 

By the side of the info table, you can see the **top cardinalities** table that shows the first ten values by order of frequency. In addition, the function returns a **barplot** of the cardinalities frequencies. The **plot_categorical** function raises the message ***too many unique values*** instead of the plot if the variable has more than 1000 unique values and removes the x-axis ticks if the variable has more than 50 unique values. 

In the **plot_categorical** function, it's not mandatory to use pandas "object" type variables, but it's strictly recommended.

### Plot numerical variables 

```python
eda.plot_numerical(data, col: list[str], bins: int = 50)
```
Parameters:

* **data**: A pandas dataframe.
* **col**: A list of string containing the names of columns to plot.
* **bins**: Number of bins to use in the histogram plot. 

Like the **plot_categorical**, the function returns a sequence of tables and plots. For every variables the **plot_quantitative** function produce an **info** table that contains the information about: 

* Count of rows not nan
* Mean
* Std
* Min
* 25%
* 50%
* 75%
* Max
* Number of unique values 
* Value of the skew 

In addition, the function returns an histogram with an estimated density + a boxplot. In the **plot_quantitative** function, it's mandatory to pass numerical variables to plot the histogram and estimate the density of the distribution. 

### View cardinalities of variables

```python
eda.view_cardinality(data, col: list[str])
```

Parameters:

* **data**: A pandas dataframe.
* **col**: A list of strings containing the names of columns for which we want to show the number of unique values.

The function especially helps study the cardinalities of the categorical variables. In case the variables present high cardinalities values. We need to reduce these values or drop the variable.
 
In addition, seeing low cardinalities values in numerical variables can be a clue for the necessity to convert a numerical variable into a categorical with the **num_to_categorical** function.

### Modify the cardinalities of a variable

```python 
eda.modify_cardinality(data, col: list[str], threshold: list[int])
```
Parameters:

* **data**: A pandas dataframe.
* **col**: A list of strings containing the names of columns for which we want to modify the cardinalities.
* **threshold**: A list of integer values containing the threshold values for every variable.

All the cardinalities that have a total count lower than the threshold indicated in the function are grouped into a new unique value called: Other. 

The function returns a pandas dataframe with the cardinalities of the columns selected modified. 

### Distribution study of a numerical variable

```python
eda.num_variable_study(data, col:str, bins: int = 50, epsilon: float = 0.0001, theory: bool = False)
```
Parameters: 

* **data**: A pandas dataframe.
* **col**: The name of the dataframe column to study.
* **bins**: The number of bins used by the histograms. By default bins=50.
* **epsilon**: A constant for handle non strictly positive variables. By default epsilon = 0.0001
* **theory**: A boolean value for displaying insight into the transformations applied

The function displays the following transformations of the variable **col** passed:

* $log(x)$
* $\sqrt(x)$
* $x^2$
* Box-cox
* $1/x$

If a variable with zeros or negative values is passed, the function shows results based on the original data transformed to be strictly positive. 

* In case of zeros, data is transformed as: $\begin{cases}
    x_i = \epsilon,& \text{if } x_i = 0\\
    x_i,              & \text{otherwise}
\end{cases}$. 

* In case of negative values, data are transformed as: $x_i = x_i + |min(x)|\cdot\epsilon$.


### Pearson's correlation matrix

```python 
eda.correlation_pearson(data, threshold: float = 0.)
```
Parameters: 

* **data**: A pandas dataframe.
* **threshold**: Only the correlation values higher than the threshold are shown in the matrix. A floating value set by default to 0. 

### Correlation matrix for categorical columns 

```python
eda.correlation_categorical(data)
```

Parameters:

* **data**: A pandas dataframe.

The function performs the Chi-Square Test of Independence between categorical variables of the dataset. 

### Phik Correlation matrix

```python
eda.correlation_phik(data, theory: bool = False)
```

Parameters:

* **data**: A pandas dataframe.
* **theory**: A boolean value for displaying insight into the theory of the $\phi_k$ correlation index. By default is set to False. 

[Link to the paper](https://arxiv.org/pdf/1811.11440.pdf)


### Interaction 

```python 
eda.interaction(data)
```

Parameters:

* **data**: A pandas dataframe.

The function display an interactive plot for analysing relationships between numerical columns with a scatterplot.

### Inspection

```python 
eda.inspection(data, threshold: int = 10, bins: int = 50, figsize: tuple[float, float] = (6., 4.))
```

Parameters:

* **data**: A pandas dataframe.
* **threshold**: A value for determining the maximum number of distinct cardinalities the target variable can have. By default is set to 10. 
* **bins**: The number of bins used by the histograms. By default bins=50.
* **figsize**: A tuple to determine the plot size.

The function displays an interactive plot for analysing the distribution of a variable based on the distinct cardinalities of the target variable. 


### Split and scaling 

```python 
eda.split_and_scaling(data, target: str)
```

Parameters: 

* **data**: A pandas dataframe.
* **target**: The response variable column name.

The function returns two pandas dataframes:

* The regressor matrix $X$ contains all the predictors for the model. 
* The series $y$ contains the values of the response variable.

In addition, the function applies a step of standard scaling on the numerical columns of the $X$ matrix.


## TODO 

* Finishing the documentation. 
* Add the xgboost model, PCA regression and other methods for studying the goodness of fit of the other models.
* Add the classification class to the package.
* Ensamble regressor/classifier method. 