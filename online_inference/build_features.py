import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_continuous_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", StandardScaler()),
        ]
    )
    return num_pipeline


def build_transformer(features_by_type) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                features_by_type.categorical_features,
            ),
            (
                "continuous_pipeline",
                build_continuous_pipeline(),
                features_by_type.categorical_features,
            ),
        ]
    )
    return transformer


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.fit_transform(df)
