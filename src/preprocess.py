from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

NUMERIC_FEATURES = [
    "LineOfCode",
    "LargestLineLength",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfSelfRef",
    "NoOfExternalRef",
    "DomainAgeMonths",
    "NoOfImage",
]

CATEGORICAL_FEATURES = [
    "Robots",
    "IsResponsive",
    "Industry",
    "HostingProvider",
]


def _log1p_safe(x: np.ndarray) -> np.ndarray:
    """
    Safe log1p:
    - Replace negative values with 0
    - Apply log1p
    - Replace any NaN from invalid operations with 0
    """
    x = np.where(x < 0, 0, x)
    out = np.log1p(x)
    out = np.where(np.isnan(out), 0, out)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(_log1p_safe, validate=False)),
            ("fillna", SimpleImputer(strategy="median")),  # extra safety
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_X_y(df: pd.DataFrame):
    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in dataframe as target.")

    df = df.dropna(subset=["label"]).copy()
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])
    return X, y
