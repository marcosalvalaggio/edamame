import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn
from IPython.display import display, Markdown, HTML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import scipy as sp
from itertools import product
import phik
from ipywidgets import interact
from edamame.eda.tools import dataframe_review
from typing import Tuple, Union, List

# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



def __display_side_by_side(dfs: List, captions: List) -> None:
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))



def dimensions(data: pd.DataFrame) -> None:
    """
    The function displays the number of rows and columns of a pandas dataframe passed. 

    Args:
        data (pd.Dataframe): A pandas DataFrame passed in input.

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
        >>> eda.dimensions(df)
    """
    # dataframe control step 
    dataframe_review(data)
    # ---
    dim = f'Rows: {data.shape[0]}, Columns: {data.shape[1]}'
    display(Markdown(dim))



def describe_distribution(data: pd.DataFrame) -> None:
    """
    The function display the result of the describe() method applied to a pandas dataframe, divided by numerical and object columns.
    
    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.

    Returns: 
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
        >>> eda.describe_distribution(df)
    """
    # dataframe control step 
    dataframe_review(data)
    string = '### Quantitative columns'
    display(Markdown(string))
    display(data.describe())
    string = '### Categorical columns'
    display(Markdown(string))
    display(data.describe(include=["O"]))



def identify_types(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    The function display the result of the dtypes method.
    
    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.

    Returns:
        Tuple[List[str], List[str]]: A tuple contains a list with the numerical columns and a list with the categorical/object column.

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4]})
        >>> quant_col, qual_col = eda.identify_types(data)
    """
     # dataframe control step 
    dataframe_review(data)
    # display types 
    types = pd.DataFrame(data.dtypes)
    types.columns = ['variable type']
    display(Markdown(types.to_markdown()))
    # quantitative variables columns 
    types = data.dtypes
    quant_col = types[types != 'object']
    quant_col = list(quant_col.index)
    # categorical variables columns
    qual_col = types[types == 'object']
    qual_col = list(qual_col.index)
     # return step 
    return quant_col, qual_col



def num_to_categorical(data: pd.DataFrame, col: List[str]) -> pd.DataFrame:
    """
    The function returns a dataframe with the columns transformed into an "object". 

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of strings containing the names of columns to convert.

    Returns:
        pd.DataFrame: Dataframe with numerical columns passed converted to categorical.

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['0', '1', '0', '1'], 'value': [1, 2, 3, 4]})
        >>> df = eda.num_to_categorical(df, col=["category"])
    """
    # dataframe check
    dataframe_review(data)
    # convert 
    data = data.copy()
    data[col] = data[col].astype(object)
    return data



def missing(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    The function display the following elements:

    - A table with the percentage of NA record for every column.
    - A table with the percentage of **0** as a record for every column.
    - A table with the percentage of duplicate rows.
    - A list of lists that contains the name of the numerical columns with NA, the name of the categorical columns with NA and the name of the columns with 0 as a record. 

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
   
    Returns: 
        Tuple[List[str], List[str], List[str]]: A Tuple that contains the name of the numerical columns with NA, the name of the categorical columns with NA and the name of the columns with 0 as a record.
    
    Example:
        >>> import edamame.eda as eda
        >>> import pandas as pd
        >>> import numpy as np 
        >>> df = pd.DataFrame({'category': ['0', '1', '0', '1', np.nan], 'value': [1, 2, 3, 4, np.nan]})
        >>> nan_quant, nan_qual, zero_col = eda.missing(df)
    """
    # dataframe control step 
    dataframe_review(data)
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
    display(Markdown(string))
    info_table = {'Row': num_row, 'Col': num_col, 'Rows without NaN': num_rows_wnan, 'Numerical variables': num_quant_col, 'Categorical variables': num_qual_col}
    info_table = pd.DataFrame(data=info_table, index = ['0'])
    display(Markdown(info_table.to_markdown(index=False)))
    # ----------------------------- #
    # null or blank values in columns 
    # ----------------------------- #
    string = '### Check blank, null or empty values'
    display(Markdown(string))
    nan = pd.Series((data.isnull().mean()*100),name ='%')
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
    string = '### Check zeros'
    display(Markdown(string))
    zero = pd.Series((data == 0).mean()*100,name ='%')
    zero.index.name = 'columns'
    zero = zero[zero>0]
    display(Markdown(zero.to_markdown()))
    zero_col = list(zero.index)
    # ----------------------------- #
    # duplicates rows
    # ----------------------------- #
    string = '### Check duplicates rows'
    display(Markdown(string))
    dupli = pd.Series(data.duplicated(keep=False).sum().mean()*100,name ='%')
    display(Markdown(dupli.to_markdown(index=False))) 
    # ----------------------------- #
    # summary 
    # ----------------------------- #
    string = '### SUMMARY table'
    display(Markdown(string))
    zero_row = np.zeros((2,data.shape[1]))
    summary = pd.DataFrame(zero_row, index = ["nan", "zero"])
    summary.columns = data.columns
    # values of nan
    summary.loc['nan',nan_col] = nan.values
    summary.loc['zero',zero_col] = zero.values
    summary = summary/100
    # background color refactor
    def highlight_cells(x):
        summary = x.copy()
        #set default color
        #set particular cell colors
        summary.loc['nan',nan_col] = 'background-color: red'
        summary.loc['zero',zero_col] = 'background-color: orange'
        return summary 
    summary = summary.style.apply(highlight_cells, axis=None).format("{:.2%}")
    display(summary)
    print("\n")
    return nan_quant, nan_qual, zero_col



# missing_cal = np.nan or 0 or other values
def handling_missing(data: pd.DataFrame, col: List[str], missing_val: Union[float, int] = np.nan, method: List[str] = []) -> pd.DataFrame:
    """
    The function returns a pandas dataframe with the columns selected modified to handle the NaN values. It's easy to use after the execution of the missing function.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of the names of the dataframe columns to handle.
        missing_val (Union[float, int]): The value that represents the NA in the columns passed. By default is equal to np.nan but can be set as other value like 0.
        method (List[str]): A list of the names of the methods (mean, median, most_frequent, drop) applied to the columns passed. By default, if nothing was indicated, the function applied the most_frequent method to all the columns passed. Indicating fewer methods than the names of the columns leads to an autocompletion with the most_frequent method.

    Returns:
        pd.DataFrame: Return the processed dataframe

    Example:
        >>> import edamame.eda as eda
        >>> import pandas as pd
        >>> import numpy as np 
        >>> df = pd.DataFrame({'category': ['0', '1', '0', '1', np.nan], 'value': [1, 2, 3, 4, np.nan], 'value_2': [-1,-2,0,0,0]})
        >>> nan_quant, nan_qual, zero_col = eda.missing(df)
        >>> df = eda.handling_missing(df, col = nan_quant, missing_val = np.nan, method = ['mean']*len(nan_quant)) # handle NaN for numerical columns
        >>> df = eda.handling_missing(df, col = nan_qual, missing_val=np.nan, method=['most_frequent']*len(nan_qual)) # handle NaN for categorical columns
        >>> df = eda.handling_missing(df, col = zero_col, missing_val=0, method=['mean']*len(zero_col)) # handle O for columns with too many zeros 
    """
    # dataframe control step 
    dataframe_review(data)
    data = data.copy()
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

    return data



def drop_columns(data: pd.DataFrame, col: List[str]):
    """
    The function returns a pandas dataframe with the columns selected dropped.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of strings containing the names of columns to drop. 

    Returns:
        pd.DataFrame

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4], 'type': [0,1,0,1]})
        >>> df = eda.drop_columns(df, col=["category", "type"])
    """
    # dataframe control step 
    dataframe_review(data)
    for _,colname in enumerate(col):
        data = data.drop(colname, axis=1)
    return data



def plot_categorical(data: pd.DataFrame, col: List[str]) -> None:
    """
    The function returns a sequence of tables and plots.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of string containing the names of columns to plot.

    Returns:
        None
    
    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4], 'type': [0,1,0,1]})
        >>> num_col, qual_col = eda.variables_type(df)
        >>> eda.plot_categorical(df, qual_col)
    """
    # dataframe check 
    dataframe_review(data)
    for _, col in enumerate(col):
        # title
        string = '### ' + col
        display(Markdown(string))
        # info table
        df = pd.DataFrame(data[col].describe())
        # cardinality table 
        df_c = pd.DataFrame(data[col].value_counts())
        df_c = df_c.head(10)
        __display_side_by_side([df, df_c], ['Info', 'Top cardinalities'])
        print('\n')
        # plot
        if len(data[col].value_counts().index) > 1000:
            string = '***too many unique values***'
            print('\n')
            display(Markdown(string))
        elif len(data[col].value_counts().index) > 50:
            fig = plt.figure(figsize = (8, 4))
            plt.bar(data[col].value_counts().index, data[col].value_counts())
            plt.xticks([''])
            plt.xticks(rotation = 90)
            plt.ylabel(col)
            plt.show()
        else:
            fig = plt.figure(figsize = (8, 4))
            plt.bar(data[col].value_counts().index, data[col].value_counts())
            plt.xticks(rotation = 90)
            plt.ylabel(col)
            plt.show()
        print('\n')



def plot_numerical(data: pd.DataFrame, col: List[str], bins: int = 50) -> None:
    """
    The function returns a sequence of tables and plots.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of string containing the names of columns to plot.
        bins (int): Number of bins to use in the histogram plot. 

    Returns:
        None
    
    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4], 'type': [0,1,0,1]})
        >>> num_col, qual_col = eda.variables_type(df)
        >>> eda.plot_numerical(df, num_col, bins = 100)
    """
    # dataframe check 
    dataframe_review(data)
    for _, col in enumerate(col):
        # title
        string = '### ' + col
        display(Markdown(string))
        # info table
        df = pd.DataFrame(data[col].describe())
        # add unique value
        unique = len(set(data[col]))
        df.loc[len(df.index)] = [unique]
        df.rename(index={8:'unique'},inplace=True)
        # add skewness value
        sk = data[col].skew()
        df.loc[len(df.index)] = [sk]
        df.rename(index={9:'skew'},inplace=True)
        __display_side_by_side([df], ['Info'])
         # plot 
        fig = plt.figure(figsize = (8, 4))
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        # assigning a graph to each ax
        sn.boxplot(x = data[col], ax=ax_box)
        sn.histplot(x = data[col], ax=ax_hist, bins=bins,kde=True)
        # Remove x axis name for the boxplot
        ax_box.set(xlabel='')
        plt.show()



def view_cardinality(data: pd.DataFrame, col: List[str]) -> None:
    """
    The function especially helps study the cardinalities of the categorical variables.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of strings containing the names of columns for which we want to show the number of unique values.  

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B'], 'value': [1, 2, 3, 4], 'type': [0,1,0,1]})
        >>> num_col, qual_col = eda.variables_type(df)
        >>> eda.view_cardinality(df, qual_col)
    """
    dataframe_review(data)
    # dataframe of the cardinalities 
    cardinality = pd.DataFrame()
    cardinality['columns'] = col
    cardinality['cardinality'] = [data[col[i]].value_counts().count() for i in range(len(col))]
    display(Markdown(cardinality.to_markdown(index=False)))



def modify_cardinality(data: pd.DataFrame, col: List[str], threshold: List[int]) -> pd.DataFrame:
    """
    The function returns a pandas dataframe with the cardinalities of the columns selected modified.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (List[str]): A list of strings containing the names of columns for which we want to modify the cardinalities.
        threshold (List[int]): A list of integer values containing the threshold values for every variable.

    Returns:
        pd.DataFrame
    
    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'type': [0,1,0,1,0,1]})
        >>> df = eda.modify_cardinality(data_cpy, col = ['category'], threshold=[3])
    """
    # dataframe check 
    dataframe_review(data)
    data = data.copy()
    # parameters check
    if len(col) != len(threshold):
        raise ValueError("You must pass the same number of values in the [col] parameter and in the [thresholds] parameter")
    else: 
        pass
    # dataframe of old cardinalities 
    cardinality = pd.DataFrame()
    cardinality['columns'] = col
    cardinality['old_cardinalities'] = [data[col[i]].value_counts().count() for i in range(len(col))]
    #display(Markdown(cardinality.to_markdown()))
    for i, colname in enumerate(col):
        listCat = data[colname].value_counts()
        listCat = list(listCat[listCat >= threshold[i]].index)
        data.loc[~data.loc[:,colname].isin(listCat), colname] = "Other"
    # add new cardinalities 
    cardinality['new_cardinalities'] = [data[col[i]].value_counts().count() for i in range(len(col))]
    # display
    display(Markdown(cardinality.to_markdown(index=False)))
    # return step 
    return data



# correlation coefficients for numerical variables (Pearson, Kendall, Spearman)
def correlation_pearson(data: pd.DataFrame, threshold: float = 0.) -> None:
    """
    The function performs the Pearson's correlation between the columns pairs. 

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        threshold (float): Only the correlation values higher than the threshold are shown in the matrix. A floating value set by default to 0. 

    Returns:
        None

    Example: 
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'type': [0,1,0,1,0,1]})
        >>> num_col, qual_col = eda.variables_type(df)
        >>> eda.correlation_pearson(df, num_col)
    """
    # dataframe check 
    dataframe_review(data)
    # title 
    string = "### Pearson's correlation matrix"
    display(Markdown(string))
    # correlation matrix 
    corr_mtr = data.corr()
    # graph style 
    if threshold == 0.: 
        corr_mtr = corr_mtr.style.background_gradient()
        display(corr_mtr)
    else: 
        corr_mtr = corr_mtr[(corr_mtr.iloc[:,:]>threshold) | (corr_mtr.iloc[:,:]<-threshold)]
        display(corr_mtr)



def correlation_categorical(data: pd.DataFrame) -> None:
    """
    The function performs the Chi-Square Test of Independence between categorical variables of the dataset. 

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'category_2': ['A2', 'A2', 'B2', 'B2', 'A2', 'B2']})
        >>> eda.correlation_categorical(df)
    """
    # dataframe check 
    dataframe_review(data)
    # title 
    string = '### $\chi^2$ test statistic $p$-values'
    display(Markdown(string))
    # list of categorical columns 
    types = data.dtypes
    qual = types[types == 'object']
    qual = list(qual.index)
    # pairs of variable 
    var_prod = list(product(qual, qual, repeat=1))
    # calculate chisq p-value 
    result = []
    for i in var_prod:
        if i[0] != i[1]:
            crosstab = pd.crosstab(data[i[0]], data[i[1]])
            chisq = list(sp.stats.chi2_contingency(crosstab))
            pval = np.round(chisq[1], decimals=3)
            result.append((i[0],i[1],pval))
    # create the matrix 
    chi_test_output = pd.DataFrame(result, columns = ['var1', 'var2', 'coeff'])
    chi_test_output = chi_test_output.pivot(index='var1', columns='var2', values='coeff')
    chi_test_output.columns.name = None
    chi_test_output.index.name = None
    # style
    def highlight_pvalue(s):
        is_rej = s < 0.05
        return ['background-color: orange' if i else 'background-color:' for i in is_rej]
    chi_test_output = chi_test_output.style.apply(highlight_pvalue)
    # display 
    display(chi_test_output)



# https://towardsdatascience.com/phik-k-get-familiar-with-the-latest-correlation-coefficient-9ba0032b37e7
def correlation_phik(data: pd.DataFrame, theory: bool = False) -> None:
    """
    Paper link: https://arxiv.org/pdf/1811.11440.pdf

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        theory (bool): A boolean value for displaying insight into the theory of the Phik correlation index. By default is set to False.

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'category_2': ['A2', 'A2', 'B2', 'B2', 'A2', 'B2']})
        >>> eda.correlation_phik(df, theory=True)
    """
    # dataframe check 
    dataframe_review(data)
    # title 
    string = '### $\phi_K$ correlation matrix'
    display(Markdown(string))
    # interval columns 
    types = data.dtypes
    quant = types[types != 'object']
    quant = list(quant.index)
    interval_cols = quant
    # call method (if categorical columns has many cardinalities slow down the execution)
    phik_overview = data.phik_matrix(interval_cols = interval_cols)
    phik_overview = phik_overview.style.background_gradient().format("{:.2}")
    # display
    display(phik_overview)
    if theory == True:
        string = '* the calculation of $\phi_K$ is computationally expensive'
        string2 = '* no indication of direction'
        string3 = '* no closed-form formula'
        string4 = '* when working with numeric-only variables, other correlation coefficients will be more precise, especially for small samples.'
        string5 = '* it is based on several refinements to Pearson’s $\chi^2$ contingency test'
        display(Markdown(string),Markdown(string2),Markdown(string3), Markdown(string4), Markdown(string5))
    else:
        pass



def __num_plot(data: pd.DataFrame, col: str, bins: int) -> None:
    # figure dim
    plt.figure(figsize = (22, 12))
    # original
    plt.subplot(2,3,1)
    sn.histplot(x = data,kde=True, bins = bins)
    plt.title('$x$')
    # log price
    plt.subplot(2,3,2)
    sn.histplot(x = np.log(data),kde=True, bins = bins)
    plt.title('$log(x)$')
    # square root
    plt.subplot(2,3,3)
    sn.histplot(x = np.sqrt(data), kde=True, bins = bins)
    plt.title('$\mathregular{\sqrt{x}}$')
    # square 
    plt.subplot(2,3,4)
    sn.histplot(x = data**2, kde=True, bins = bins)
    plt.title('$\mathregular{{x}^2}$')
    # box-cox (with lambda=none, array must be positive)
    plt.subplot(2,3,5)
    x = sp.stats.boxcox(data)
    lmbda = x[1]
    sn.histplot(x = x[0], kde=True, bins = bins)
    plt.xlabel(col)
    plt.title(f'Box-Cox with $\lambda$ = {lmbda:.3f}')
    # reciprocal 
    plt.subplot(2,3,6)
    sn.histplot(x = 1/data, kde=True, bins = bins)
    plt.title('$1/x$')
    plt.show()



# useful tests for normality check: shapiro-wilk, wilcoxon, qq-plot
def num_variable_study(data: pd.DataFrame, col: str, bins: int = 50, epsilon: float = 0.0001, theory: bool = False) -> None:
    """
    The function displays the following transformations of the variable col passed: log(x), sqrt(x), x^2, Box-cox, 1/x

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        col (str): The name of the dataframe column to study.
        bins (int): The number of bins used by the histograms. By default bins=50.
        epsilon (float): A constant for handle non strictly positive variables. By default epsilon = 0.0001 
        theory (bool): A boolean value for displaying insight into the transformations applied.

    Returns:
        None
    
    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5]})
        >>> eda.num_variable_study(df, 'value', bins = 50, theory=True)
    """
    # dataframe chack step 
    dataframe_review(data)
    # response variable check step 
    if data[col].dtypes == 'O':
        raise TypeError('you must pass a quantitative variable')
    else:
        pass
    # check
    if data[data[col] < 0].shape[0] > 0:
        string = '## Variable with negative values'
        display(Markdown(string))
        string = "* applied the transformation $x^{'}=x-min(x)+\epsilon$"
        display(Markdown(string))
        x = data[col]
        x = x + abs(min(x))+epsilon
        __num_plot(data = x, col = col, bins = bins)
    elif data[data[col] == 0].shape[0] > 0:
        string = '## Variable with zeros values'
        display(Markdown(string))
        x = data[col]
        x[x==0] = epsilon
        __num_plot(data = x, col = col, bins = bins)
    else:
        string = '## Strickt positive variable'
        display(Markdown(string))
        x = data[col]
        __num_plot(data = x, col = col, bins = bins)
    if theory == True:
        # theory behind transformation 
        string = '## Effects of transformations:'
        display(Markdown(string))
        string = '### log transformation'
        display(Markdown(string))
        string = '* positive effect on right-skewed distributions and de-emphasize outliers'
        string2 = '* gets worse when applied to distributions left-skewed or already normal'
        display(Markdown(string),Markdown(string2))
        string = '### Square root transformation'
        display(Markdown(string))
        string = '* normalizing effect on right-skewed distributions, it is weaker than the logarithm and cube root'
        string2 = '* variables with a left skew will become worst after a square root transformation.'
        string3 = '* high values get compressed and low values become more spread out'
        display(Markdown(string),Markdown(string2),Markdown(string3))
        string = '### Square transformation'
        display(Markdown(string))
        string = '* used to reduce left skewness'
        string2 = '* gets worse when applied to distributions without skewness'
        display(Markdown(string),Markdown(string2))
        string = '### Box-Cox'
        display(Markdown(string))
        string = '* if $\lambda$ is a non-zero number, then the transformed variable may be more difficult to interpret than if we simply applied a log transform.'
        string2 = '* works well for left and right skewness'
        string3 = '* only works for positive data'
        display(Markdown(string),Markdown(string2),Markdown(string3))
        string = '### Reciprocal'
        display(Markdown(string))
        string = '* It can not be applied to zero values'
        string2 = '* The reciprocal reverses order among values of the same sign: largestbecomes smallest, etc.'
        display(Markdown(string),Markdown(string2))
    else: 
        pass



def interaction(data: pd.DataFrame) -> None:
    """
    The function display an interactive plot for analysing relationships between numerical columns with a scatterplot.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5]})
        >>> eda.interaction(df)
    """
    # dataframe check 
    dataframe_review(data)
    # scatterplot function 
    @interact
    def scatter(column1 = list(data.select_dtypes('number').columns), 
                column2 = list(data.select_dtypes('number').columns)):
        # scatterplot
        sn.jointplot(x=data[column1], y=data[column2], kind='scatter')
        plt.show()



def inspection(data: pd.DataFrame, threshold: int = 10, bins: int = 50, figsize: Tuple[float, float] = (6., 4.)) -> None:
    """
    The function displays an interactive plot for analysing the distribution of a variable based on the distinct cardinalities of the target variable.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        threshold (int): A value for determining the maximum number of distinct cardinalities the target variable can have. By default is set to 10. 
        bins (int): The number of bins used by the histograms. By default bins=50.
        figsize (Tuple[float, float]): A tuple to determine the plot size.

    Returns:
        None

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5]})
        >>> eda.inspection(df)
    """
    # dataframe check 
    dataframe_review(data)
    # plot step
    @interact
    def inspect(column = list(data.columns), target = list(data.columns)):
        # useful info
        title = ' ### Variable type info:'
        display(Markdown(title))
        column_type = str(data[column].describe().dtype)
        target_type = str(data[target].describe().dtype)
        string = f'**column**: {column_type}, **target**: {target_type}'
        display(Markdown(string))
        # check column variable type
        if len(data[target].unique()) > threshold:
            print('too many unique values in the target variable')
        # check column variable type
        elif data[column].dtypes == 'O':
            #barplot 
            plt.figure(figsize=figsize)
            sn.countplot(x=data[column], hue=data[target])
            plt.xticks(rotation=90)           
            plt.show()
        else:
            # histplot
            plt.figure(figsize=figsize)
            try:
                sn.histplot(x=data[column], hue=data[target], kde = True, bins = bins)
            except:
                sn.histplot(x=data[column], hue=data[target], bins = bins)
            plt.show()



def split_and_scaling(data: pd.DataFrame, target: str, minmaxscaler: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    The function returns two pandas dataframes: 

        * The regressor matrix X contains all the predictors for the model.
        * The series y contains the values of the response variable.

    In addition, the function applies a step of standard scaling on the numerical columns of the X matrix.

    Args:
        data (pd.DataFrame): A pandas DataFrame passed in input.
        target (str): The response variable column name.
        minmaxscaler (bool): Select the type of scaling to apply to the numerical columns. 
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Return the regression matrix and the target column. 

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'target': ['A2', 'A2', 'B2', 'B2', 'A2', 'B2']})
        >>> X, y = eda.split_and_scaling(df, 'target')
    """
    # dataframe check 
    dataframe_review(data)
    # split step
    y = data[target]
    X = data.drop(target, axis=1)
    # scaling quantitative variables 
    types = X.dtypes
    quant = types[types != 'object']
    quant_columns = list(quant.index)
    if minmaxscaler:
        scaler = MinMaxScaler()
    else: 
        scaler = StandardScaler()
    scaler.fit(X[quant_columns])
    X[quant_columns] = scaler.transform(X[quant_columns])
    # return step 
    return X,y