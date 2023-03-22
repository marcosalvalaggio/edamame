import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# ----------------- #
# load model 
# ----------------- #
def load_model(path: str):
    """
    The function load the model saved previously in the pickle format.

    Parameters
    ----------
    path: str
        Path to the model saved in .pkl

    Return
    ----------
        model loaded 
    """
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model 


# ----------------- #
# setup data for train 
# ----------------- #
def setup(X, y, dummy: bool = False, seed: int = 42, size: float = 0.25):
    """
    The function returns the following elements: X_train, y_train, X_test, y_test.

    Parameters
    ----------
    X: pandas.DataFrame
        The model matrix X (features matrix)
    y: pandas.DataFrame
        The target variable 
    dummy: bool
        If False, the function produces the OHE. If True, the dummy encoding
    seed: int
        Random seed to apply at the train_test_split function
    size: float
        Size of the test dataset 

    Return
    ----------
    pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame
    """
    # split dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    X_train = pd.get_dummies(data=X_train, drop_first=dummy)
    X_test = pd.get_dummies(data=X_test, drop_first=dummy)
    return X_train, y_train, X_test, y_test


# --------------------- #
# scale X dataset
# --------------------- #
def scaling(X, minmaxscaler: bool = False):
    """
    The function returns the normalised matrix.

    Parameters
    ----------
    X: pandas.DataFrame
        The model matrix X/X_train/X_test
    minmaxscaler: bool
        Select the type of scaling to apply to the numerical columns. 

    Return
    ----------
    pandas.DataFrame
    """
    # dataframe check 
    dataframe_review(X)
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
    return X


# ----------------- #
# OHE check
# ----------------- #
def dummy_control(data):
    """
    The function checks if the Pandas dataframe passed is encoded with dummy or OHE. 
    
    Parameters
    ----------
    data: pandas.DataFrame 

    Raises
    ----------
    TypeError
        Dataframe with non-numerical columns
    """
    types = data.dtypes
    qual_col = types[types == 'object']
    if len(qual_col) != 0:
        raise TypeError('dataframe with non-numerical columns')
    else:
        pass


# ----------------- #
#  pandas dataframe check
# ----------------- #
def dataframe_review(data) -> None:
    """
    The function checks if the object passed is a Pandas dataframe.

    Parameters
    ----------
    data: A pandas dataframe 

    Raises
    ----------
    TypeError
        The data loaded is not a DataFrame
    """
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')
    