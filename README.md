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
eda.identify_types(data, col: list[str])
```

passing a dataframe and a list with columns name, the function transforms the types of the columns into "object". This operation can help convert numerical columns we know to be categorical. 


### Missing data

```python
eda.missing(data)
```



## TODO 

* Finishing the documentation 
* Add the xgboost model, PCA regression and other methods for studying the goodness of fit of the other models
* Add the classification part to the package 