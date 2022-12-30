import pickle
from sklearn.model_selection import train_test_split
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
    