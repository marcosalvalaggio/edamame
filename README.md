# Installation

To install the package,

```
pip install edamame
```

edamame functions work correctly inside a **.ipynb** file. 

```python
import edamame as eda
```


## Exploratory data analysis functions


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
