import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn
import IPython as ip
import sklearn.impute
import plotly.subplots as sub
import plotly.graph_objs as go
# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



# --------------------- #
# Dataframe type control
# --------------------- #
def dataframe_review(data):
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')


# --------------------- #
# Dataframe dimensions
# --------------------- #
def dimensions(data):
    # dataframe control step 
    dataframe_review(data)
    # ---
    dim = f'Rows: {data.shape[0]}, Columns: {data.shape[1]}'
    ip.display.display(ip.display.Markdown(dim))

# test
#dimensions(data)

# --------------------- #
# full describe function 
# --------------------- #
def describe_distribution(data):
    # dataframe control step 
    dataframe_review(data)
    # ---
    string = '### Quantitative columns'
    ip.display.display(ip.display.Markdown(string))
    ip.display.display(data.describe())
    string = '### Categorical columns'
    ip.display.display(ip.display.Markdown(string))
    ip.display.display(data.describe(include=["O"]))

# test
#describeDistribution(data)


# --------------------- #
# variables types identifier 
# --------------------- #
def variables_type(data):
     # dataframe control step 
    dataframe_review(data)
    # quantiative variables columns 
    types = data.dtypes
    quant_col = types[types != 'object']
    quant_col = list(quant_col.index)
    # categorical variables columns
    qual_col = types[types == 'object']
    qual_col = list(qual_col.index)
     # return step 
    return [quant_col, qual_col]

# test
#quant_col, qual_col = variables_type(data)
#print(quant_col)
#print(qual_col)

# --------------------- #
# change variables types identifier 
# --------------------- #
def change_variable_type(data, col: list[str]) -> list[list]:
    quant_col, qual_col = variables_type(data)
    for _,colname in enumerate(col):
        if colname in quant_col:
            quant_col.remove(colname)
            qual_col.append(colname)
        elif colname in qual_col:
            qual_col.remove(colname)
            quant_col.append(colname)
        else: 
            pass
    return [quant_col, qual_col]

# test 
#quant_col, qual_col = change_variable_type(data = train_df, col=['Pclass','SibSp'])
#print(quant_col)
#print(qual_col)




# --------------------- #
# missing, zeros and duplicates 
# --------------------- #
def missing(data):
    # dataframe control step 
    dataframe_review(data)
    # ---
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
    string = '### INFO table'
    ip.display.display(ip.display.Markdown(string))
    info_table = {'Row': num_row, 'Col': num_col, 'Rows without NaN': num_rows_wnan, 'Quantitative variables': num_quant_col, 'Categorical variables': num_qual_col}
    info_table = pd.DataFrame(data=info_table, index = ['0'])
    ip.display.display(ip.display.Markdown(info_table.to_markdown()))
    # ----------------------------- #
    # null or blank values in columns 
    # ----------------------------- #
    string = '### Check blank, null or empty values'
    ip.display.display(ip.display.Markdown(string))
    nan = pd.Series((data.isnull().mean()),name ='%')
    nan.index.name = 'columns'
    nan = nan[nan>0]
    ip.display.display(ip.display.Markdown(nan.to_markdown()))
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
    string = '### Check zeros'
    ip.display.display(ip.display.Markdown(string))
    zero = pd.Series((data == 0).mean(),name ='%')
    zero.index.name = 'columns'
    zero = zero[zero>0]
    ip.display.display(ip.display.Markdown(zero.to_markdown()))
    zero_col = list(zero.index)
    # ----------------------------- #
    # duplicates rows
    # ----------------------------- #
    string = '### Check duplicates rows'
    ip.display.display(ip.display.Markdown(string))
    dupli = pd.Series(data.duplicated(keep=False).sum().mean(),name ='%')
    ip.display.display(ip.display.Markdown(dupli.to_markdown())) 
    # ----------------------------- #
    # summary 
    # ----------------------------- #
    string = '### SUMMARY table'
    ip.display.display(ip.display.Markdown(string))
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
    ip.display.display(summary)
    print("\n")
    # ----------------------------- #
    # return step
    # ----------------------------- #
    return [nan_quant, nan_qual, zero_col]

# test
#nan_quant, nan_qual, zero_col = missing(data)




# --------------------- #
# handling missing and zeros
# --------------------- #
def handling_missing(data, col: list[str], missing_val = np.nan, method: list[str] = []):
    # dataframe control step 
    dataframe_review(data)
    # ---
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
            imputer = sklearn.impute.SimpleImputer(strategy=method[i], missing_values=missing_val)
            data[col[i]] = imputer.fit_transform(data[col[i]].values.reshape(-1,1))[:,0]
            #print(data.shape)
    # ----------------------------- #
    # return step
    # ----------------------------- #
    return data

# test 
#data_test = data.copy()
# quant variables
#data_test = handlingMissing(data_test, col = nan_quant, missing_val = np.nan, method = ['mean']*len(nan_quant))
# qual variables 
#data_test = handlingMissing(data_test, col = nan_qual, missing_val=np.nan, method=['most_frequent']*len(nan_qual))
# zero variables
#data_test = handlingMissing(data_test, col = zero_col, missing_val=0, method=['mean']*len(zero_col))


# --------------------- #
# fast drop columns
# --------------------- #
def drop_columns(data, col: list[str]):
    for _,colname in enumerate(col):
        data = data.drop(colname, axis=1)
    return data
# test 
#train = drop_column(train_df, col = ['Name', 'Cabin', 'PassengerId', 'Ticket'])


# --------------------- #
# plot categorical columns 
# --------------------- #
def plot_categorical(data, col: list[str]) -> None:
    # dataframe check 
    dataframe_review(data)
    # specs list 
    sp = [{'type': 'table'}, {'type': 'bar'}]
    specs = []
    for i in range(len(col)):
        specs.append(sp)
    # define figure specs
    fig = sub.make_subplots(rows=len(col), cols=2, shared_xaxes=False, horizontal_spacing=0.1, specs=specs)
    # print loop 
    for i in range(len(col)):
        # define info table
        df = pd.DataFrame(data[col[i]].describe())
        # table plot
        fig.add_trace(
            go.Table(
            header=dict(values=list(['index', df.columns[0]]),
                    #fill_color='seagreen',
                    align='center'),
            cells=dict(values=[df.index, df.iloc[:,0]],
                   #fill_color='lightcyan',
                   align='left')),
            row=i+1, col=1)
        # bar plot 
        fig.add_trace(
           go.Bar(
           x = data[col[i]].value_counts().index,
           y = data[col[i]].value_counts()), 
           row=i+1, col=2) 
    # fig dimensions 
    fig.update_layout(height=500*len(col),showlegend=False,title_text='Categorical columns')
    fig.show() 

# test 
#plot_categorical(data_test, qual_col)