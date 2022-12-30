

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

Edamame is born under the inspiration of the [pandas-profiling](https://github.com/ydataai/pandas-profiling) and [pycaret](https://github.com/pycaret/pycaret) packages. The scope of edamame is to build friendly and helpful functions for handling the EDA (exploratory data analysis) step in a dataset studied and after that train and analyze a models battery for regression or classification problems. 

## Exploratory data analysis functions

You can find an example of the EDA that uses the edamame package in the  [eda_example.ipynb](notebook/eda_example.ipynb) notebook. 

### Dimensions

a prettier version of the **.shape** method

```python
eda.dimensions(data)
```
the function displays the number of rows and columns of a pandas dataframe passed 

### Describe distribution


```python
eda.describe_distribution(data)
```

passing a dataframe the function display the result of the **.describe()** method applied to a pandas dataframe, divided by quantitative/numerical and categorical/object columns.


### Identify columns types


```python
eda.identify_types(data)
```

passing a dataframe the function display the result of the **.dtypes** method and returns a list with the name of the quantitative/numerical columns and a list with the name of the columns identified as "object" by pandas. 


### Convert numerical columns to categorical

```python
eda.num_to_categorical(data, col: list[str])
```

passing a dataframe and a list with columns name, the function transforms the types of the columns into "object". 


### Missing data

```python
eda.missing(data)
```

the function display the following elements:

* a table with the percentage of **NA** record for every column
* a table with the percentage of **0** as a record for every column
* a table with the percentage of duplicate rows
* a list of lists that contains the name of the numerical columns with **NA**, the name of the categorical columns with **NA** and the name of the columns with 0 as a record 

### Handling Missing values

```python
eda.handling_missing(data, col: list[str], missing_val = np.nan, method: list[str] = [])
```

Parameters: 

* **data**: a pandas dataframe
* **col**: a list of the names of the dataframe columns to handle
* **missing_val**: the value that represents the **NA** in the columns passed. By default is equal to **np.nan** 
* **method**: a list of the names of the methods (mean, median, most_frequent, drop) applied to the columns passed. By default, if nothing was indicated, the function applied the **most_frequent** method to all the columns passed. Indicating fewer methods than the names of the columns leads to an autocompletion with the **most_frequent** method

### Drop columns 

```python 
eda.drop_columns(data, col: list[str]):
```

Parameters:

* **data**: a pandas dataframe
* **col**: a list of strings containing the names of columns to drop 


### Plot categorical variables

```python 
eda.plot_categorical(data, col: list[str])
```
Parameters:

* **data**: a pandas dataframe
* **col**: a list of string containing the names of columns to plot
  
The function returns a sequence of tables and plots. For every variables the **plot_categorical** produce an **info** table that contains the information about: 

* the number of not nan rows 
* the number of unique values 
* the name of the value with the major frequency
* the frequence of the top unique value 

By the side of the info table, you can see the **top cardinalities** table that shows the first ten values by order of frequency. In addition, the function returns a **barplot** of the cardinalities frequencies. The **plot_categorical** function raises the message ***too many unique values*** instead of the plot if the variable has more than 1000 unique values and removes the x-axis ticks if the variable has more than 50 unique values. 

In the **plot_categorical** function, it's not mandatory to use pandas "object" type variables, but it's strictly recommended

### Plot numerical variables 

```python
eda.plot_numerical(data, col: list[str], bins: int = 50)
```
Parameters:

* **data**: a pandas dataframe
* **col**: a list of string containing the names of columns to plot
* **bins**: number of bins to use in the histogram plot 

Like the **plot_categorical**, the function returns a sequence of tables and plots. For every variables the **plot_quantitative** function produce an **info** table that contains the information about: 

* count of rows not nan
* mean
* std
* min
* 25%
* 50%
* 75%
* max
* number of unique values 
* value of the skew 

In addition, the function returns an histogram with an estimated density + a boxplot. In the **plot_quantitative** function, it's mandatory to pass numerical variables to plot the histogram and estimate the density of the distribution. 



## TODO 

* Finishing the documentation 
* Add the xgboost model, PCA regression and other methods for studying the goodness of fit of the other models
* Add the classification part to the package 