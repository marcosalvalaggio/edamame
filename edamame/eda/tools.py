import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from typing import Tuple



def load_model(path: str):
    """
    The function load the model saved previously in the pickle format.

    Args:
        path (str): Path to the model saved in .pkl

    """
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model 



def setup(X: pd.DataFrame, y: pd.DataFrame, dummy: bool = False, seed: int = 42, size: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    The function returns the following elements: X_train, y_train, X_test, y_test.

    Args: 
        X (pd.DataFrame): The model matrix X (features matrix).
        y (pd.DataFrame): The target variable.
        dummy (bool): If False, the function produces the OHE. If True, the dummy encoding.
        seed (int): Random seed to apply at the train_test_split function.
        size (float): Size of the test dataset. 

    Return:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, y_train, X_test, y_test.

    Example:
        >>> import edamame.eda as eda
        >>> df = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5], 'target': ['A2', 'A2', 'B2', 'B2', 'A2', 'B2']})
        >>> X, y = eda.split_and_scaling(df, 'target')
        >>> X_train, y_train, X_test, y_test = eda.setup(X, y)
    """
    # split dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    X_train = pd.get_dummies(data=X_train, drop_first=dummy)
    X_test = pd.get_dummies(data=X_test, drop_first=dummy)
    return X_train, y_train, X_test, y_test



def scaling(X: pd.DataFrame, minmaxscaler: bool = False) -> pd.DataFrame:
    """
    The function returns the normalised/standardized matrix.

    Args:
        X (pd.DataFrame): The model matrix X/X_train/X_test
    minmaxscaler (bool): Select the type of scaling to apply to the numerical columns. By default is setted to the StandardScaler. If minmaxscaler is set to True the numerical columns is trasfomed to [0,1].

    Return:
        pd.DataFrame

    Example:
        >>> import edamame.eda as eda
        >>> X = pd.DataFrame({'category': ['A', 'B', 'A', 'B', 'C', 'A'], 'value': [1, 2, 3, 4, 4, 5]})
        >>> X = eda.scaling(X)
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



def dummy_control(data: pd.DataFrame) -> None:
    """
    The function checks if the Pandas dataframe passed is encoded with dummy or OHE. 
    
    Args:
        data (pd.Dataframe): A pandas DataFrame passed in input.

    Raises:
        TypeError: If the input DataFrame contains non-numerical columns.

    Returns:
        None
    """
    types = data.dtypes
    qual_col = types[types == 'object']
    if len(qual_col) != 0:
        raise TypeError('dataframe with non-numerical columns')
    else:
        pass



def dataframe_review(data: pd.DataFrame) -> None:
    """
    The function checks if the object passed is a Pandas dataframe.

    Args:
        data (pd.Dataframe): A pandas DataFrame passed in input.

    Raises:
        TypeError: If the input DataFrame contains non-numerical columns.

    Returns:
        None
    """
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')
    