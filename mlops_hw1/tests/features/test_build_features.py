import random
import pandas as pd
import numpy as np
from unittest import TestCase
from collections import defaultdict
from mlops_hw1.enties.features import FeaturesByType
from mlops_hw1.features.build_features import (make_features,
                                               build_transformer)


class TestBuildFeatures(TestCase):

    @staticmethod
    def return_continuous_data():
        return [random.randint(0, 1) for _ in range(100)]

    @staticmethod
    def return_categorical_data():
        return [random.choice([0, 1, 2]) for _ in range(100)]

    def return_data_dict(self):
        data_dict = {'first_continuous_feature': self.return_continuous_data(),
                     'second_continuous_feature': self.return_continuous_data(),
                     'first_categorical_feature': self.return_categorical_data(),
                     'second_categorical_feature': self.return_categorical_data()
                     }
        return data_dict

    def test_make_features(self):
        data = pd.DataFrame.from_dict(self.return_data_dict())

        categorical_features = ['first_categorical_feature',
                                'second_categorical_feature']
        continuous_features = ['first_continuous_feature',
                               'second_continuous_feature']

        transformer = build_transformer(FeaturesByType(categorical_features=categorical_features,
                                                       continuous_features=continuous_features,
                                                       target_columns=list()))
        converted_data = make_features(transformer, data)

        self.assertEqual(8, converted_data.shape[1])

        proximity_counter = defaultdict(int)
        for column_index in range(converted_data.shape[1]):

            proximity_counter['mean_proximity_counter'] += (abs(np.mean(converted_data[:, column_index]) - 0) < 1e-9)
            proximity_counter['var_proximity_counter'] += (abs(np.var(converted_data[:, column_index]) - 1) < 1e-9)

        for key, value in proximity_counter.items():
            self.assertEqual(value, len(continuous_features))
