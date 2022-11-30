import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# --------------------- #
# Dataframe dimensions
# --------------------- #
def dimensions(data):
    if data.__class__.__name__ == 'DataFrame':
        dim = f'Rows: {data.shape[0]}, Columns: {data.shape[1]}'
        return dim
    else: 
        raise TypeError('The data loaded is not a dataframe')
# test
#dimensions(data)

# --------------------- #
# full describe function 
# --------------------- #
def describeDistribution(data):
    if data.__class__.__name__ == 'DataFrame':
        print('- Quantitative columns -')
        display(data.describe())
        print('- Categorical columns -')
        display(data.describe(include=["O"]))
    else:
        raise TypeError('The data loaded is not a dataframe')
# test
#describeDistribution(data)



# --------------------- #
# missing, zeros and duplicates 
# --------------------- #
def missing(data) -> list:
    if data.__class__.__name__ == 'DataFrame':
        # ----------------------------- #
        # info table 
        # ----------------------------- #
        # num_row
        num_row = data.shape[0]
        # num_col
        num_col = data.shape[1]
        # num rows without na
        num_rows_wnan = data.dropna().shape[0]
        # num_quant_col
        types = data.dtypes
        quant_col = types[types != 'object']
        quant_col = list(quant_col.index)
        num_quant_col = len(quant_col)
        # num_qual_col
        types = data.dtypes
        qual_col = types[types == 'object']
        qual_col = list(qual_col.index)
        num_qual_col = len(qual_col)
        # create table
        print("\nINFO table:\n")
        info_table = {'Row': num_row, 'Col': num_col, 'Rows without NaN': num_rows_wnan, 'Quantitative variables': num_quant_col, 'Categorical variables': num_qual_col}
        info_table = pd.DataFrame(data=info_table, index = ['0'])
        display(Markdown(info_table.to_markdown()))
        # ----------------------------- #
        # null or blank values in columns 
        # ----------------------------- #
        print("\nCheck blank, null or empty values:\n")
        nan = pd.Series((data.isnull().mean()),name ='%')
        nan.index.name = 'columns'
        nan = nan[nan>0]
        display(Markdown(nan.to_markdown()))
        nan_col = list(nan.index)
        # nan quantitative cols
        types = data[nan_col].dtypes
        nan_quant = types[types != 'object']
        nan_quant = list(nan_quant.index)
        # nan qualitative cols
        nan_qual = types[types == 'object']
        nan_qual = list(nan_qual.index)
        # ----------------------------- #
        # rows with zeros 
        # ----------------------------- #
        print("\nCheck zeros:\n")
        zero = pd.Series((data == 0).mean(),name ='%')
        zero.index.name = 'columns'
        zero = zero[zero>0]
        display(Markdown(zero.to_markdown()))
        zero_col = list(zero.index)
        # ----------------------------- #
        # duplicates rows
        # ----------------------------- #
        print("\nCheck duplicates rows:\n")
        dupli = pd.Series(data.duplicated(keep=False).sum().mean(),name ='%')
        display(Markdown(dupli.to_markdown())) 
        # ----------------------------- #
        # summary 
        # ----------------------------- #
        print("\nSUMMARY table:\n")
        zero_row = np.zeros((2,data.shape[1]))
        summary = pd.DataFrame(zero_row, index = ["nan", "zero"])
        summary.columns = data.columns
        # values of nan
        summary.loc['nan',nan_col] = nan.values
        summary.loc['zero',zero_col] = zero.values
        # background color refactor
        def highlight_cells(x):
            summary = x.copy()
            #set default color
            #set particular cell colors
            summary.loc['nan',nan_col] = 'background-color: red'
            summary.loc['zero',zero_col] = 'background-color: lightblue'
            return summary 
        summary = summary.style.apply(highlight_cells, axis=None).format("{:.2%}")
        display(summary)
        print("\n")
        # ----------------------------- #
        # return step
        # ----------------------------- #
        return [nan_quant, nan_qual, zero_col]
    else:
        raise TypeError('The data loaded is not a dataframe')
# test
#nan_quant, nan_qual, zero_col = missing(data)




# --------------------- #
# handling missing and zeros
# --------------------- #
def handlingMissing(data, col: list[str], missing_val = np.nan, method: list[str] = []):
    if data.__class__.__name__ == 'DataFrame':
        # ----------------------------- #
        # check method 
        # ----------------------------- #
        # possible method: mean, median, most_frequent, drop
        if len(method) == 0:
            method = ['most_frequent'] * len(col)
        elif 0 < len(method) < len(col):
            num_method = len(col) - len(method)
            add_method = ['most_frequent'] * num_method
            method.extend(add_method)
        elif len(method) > len(col):
            raise ValueError('the length of the methods list must be at least the same length as the column with missing values')
        else:
            pass
        # ----------------------------- #
        # imputation loop
        # ----------------------------- #
        for i in range(len(col)):
            if method[i] == 'drop':
                #print('traccia')
                data = data.drop(col[i], axis = 1)
                #print(data.shape)
            else:
                imputer = SimpleImputer(strategy=method[i], missing_values=missing_val)
                data[col[i]] = imputer.fit_transform(data[col[i]].values.reshape(-1,1))[:,0]
                #print(data.shape)
        # ----------------------------- #
        # return step
        # ----------------------------- #
        return data
    else:
        raise TypeError('The data loaded is not a dataframe')
# test 
#data_test = data.copy()
# quant variables
#data_test = handlingMissing(data_test, col = nan_quant, missing_val = np.nan, method = ['mean']*len(nan_quant))
# qual variables 
#data_test = handlingMissing(data_test, col = nan_qual, missing_val=np.nan, method=['most_frequent']*len(nan_qual))
# zero variables
#data_test = handlingMissing(data_test, col = zero_col, missing_val=0, method=['mean']*len(zero_col))