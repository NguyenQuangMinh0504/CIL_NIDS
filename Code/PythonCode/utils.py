from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn import preprocessing


def encode_numeric_zscore(df: DataFrame, name: str, mean=None, std=None):
    if mean is None:
        mean = df[name].mean()
    if std is None:
        std = df[name].std()
    df[name] = (df[name] - mean) / std


def encode_text_dummy(df: DataFrame, name: str):
    dummies = pd.get_dummies(df[name])
    for column in dummies.columns:
        dummy_name = "{}-{}".format(name, column)
        df[dummy_name] = dummies[column]
    df.drop(name, axis=1, inplace=True)


def encode_text_index(df: DataFrame, name: str):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_


def to_xy(df: DataFrame, target: str):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    target_type = df[target].dtype
    if target_type in (np.int64, np.int32):
        # Classification
        dummies = pd.get_dummies(df[target])
        return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
    # Regression
    return df[result].values.astype(np.float32), df[[target]].values.astype(np.float32)
