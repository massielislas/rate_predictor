import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer
from sklearn.impute import SimpleImputer

IGNORE_COL = "Region"


def load(dataset: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load dataset
    print("loading dataset:", dataset)
    data = pd.read_csv(dataset, index_col=0)
    if IGNORE_COL in data.columns.values:  # delete the Region column from the data
        del data[IGNORE_COL]
    X = data[data.columns[1:]]
    Y = data[data.columns[0]]
    return data, X, Y


def scale(method: str, data: pd.DataFrame) -> np.ndarray:
    scaler = {
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "normalize": Normalizer,
    }[method]()
    return scaler.fit_transform(data)


def impute(name: str, data:pd.DataFrame, strategy: str = "constant", fill_value: str = 0) -> np.ndarray:
    imputer = None
    if name == "simple":
        imputer = SimpleImputer(strategy=strategy)
    elif name == "fill":
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)

    if imputer:
        return imputer.fit_transform(data)
