import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin


class MeanSubtractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, features: pd.DataFrame, labels: Optional[pd.DataFrame] = None):
        return self

    def transform(self, features: pd.DataFrame, labels: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        for column in features.columns:
            try:
                curr_mean = features[column].mean()
                features[column] -= curr_mean
            except TypeError:
                pass
        return features


