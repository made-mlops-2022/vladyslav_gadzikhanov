from dataclasses import dataclass
from typing import List


@dataclass()
class FeaturesByType:
    categorical_features: List[str]
    continuous_features: List[str]
    target_columns: List[str]
