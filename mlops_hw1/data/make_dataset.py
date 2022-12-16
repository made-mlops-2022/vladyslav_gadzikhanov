import pandas as pd
from typing import Tuple
from mlops_hw1.enties.split_params import SplittingParams
from mlops_hw1.enties.features import FeaturesByType
from sklearn.model_selection import train_test_split


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_features_labels_data(data: pd.DataFrame,
                               features_by_type: FeaturesByType) -> Tuple[pd.DataFrame, pd.DataFrame]:

    features_columns = data.columns.drop(features_by_type.target_columns)
    features, labels = data[features_columns], data[features_by_type.target_columns]
    return features, labels


def split_train_test_data(features: pd.DataFrame,
                          labels: pd.DataFrame,
                          splitting_params: SplittingParams) -> Tuple[pd.DataFrame,
                                                                      pd.DataFrame,
                                                                      pd.DataFrame,
                                                                      pd.DataFrame]:

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=splitting_params.test_size,
                                                        shuffle=True,
                                                        random_state=splitting_params.random_state)
    return x_train, x_test, y_train, y_test
