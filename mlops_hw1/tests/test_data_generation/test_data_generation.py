from unittest import TestCase
from mlops_hw1.data_generation.data_generation import data_generator
from mlops_hw1.enties.features import FeaturesByType
from mlops_hw1.data.make_dataset import read_data


class TestDataGeneration(TestCase):
    @staticmethod
    def return_continuous_features():
        return ['trestbps', 'chol', 'thalach', 'oldpeak', 'age']

    @staticmethod
    def return_categorical_features():
        return ['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca']

    def test_data_generation(self):
        df = read_data('../../data/heart_cleveland_upload.csv')
        features_by_type = FeaturesByType(categorical_features=self.return_categorical_features(),
                                          continuous_features=self.return_continuous_features(),
                                          target_columns=['condition'])

        new_df = data_generator(df=df,
                                features_by_type=features_by_type)

        self.assertTupleEqual(df.shape, new_df.shape)
        self.assertSetEqual(set(df.columns), set(new_df.columns))
