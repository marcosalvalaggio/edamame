import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ----------------- #
# load model 
# ----------------- #
def load_model(path: str):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model 


# ----------------- #
# setup data for train 
# ----------------- #
def setup(X, y, dummy: bool = False, seed: int = 42, size: float = 0.25):
    # split dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=seed)
    X_train = pd.get_dummies(data=X_train, drop_first=dummy)
    X_test = pd.get_dummies(data=X_test, drop_first=dummy)
    return [X_train, X_test, y_train, y_test]


# --------------------- #
# scale X dataset
# --------------------- #
def scaling(X):
    # dataframe check 
    dataframe_review(X)
    # scaling quantitative variables 
    types = X.dtypes
    quant = types[types != 'object']
    quant_columns = list(quant.index)
    scaler = StandardScaler()
    scaler.fit(X[quant_columns])
    X[quant_columns] = scaler.transform(X[quant_columns])
    return X


# ----------------- #
# OHE check
# ----------------- #
def dummy_control(data):
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
    if data.__class__.__name__ == 'DataFrame':
        pass
    else:
        raise TypeError('The data loaded is not a DataFrame')
    