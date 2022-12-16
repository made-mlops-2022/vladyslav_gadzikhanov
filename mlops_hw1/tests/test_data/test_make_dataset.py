from unittest import TestCase
from mlops_hw1.enties.features import FeaturesByType
from mlops_hw1.enties.split_params import SplittingParams
from mlops_hw1.data.make_dataset import (read_data,
                                         split_features_labels_data,
                                         split_train_test_data)


class TestMakeDataset(TestCase):

    def setUp(self):
        self.fake_data = read_data('../../data/fake_data.csv')
        self.target_columns = ['condition']
        self.features_by_type = FeaturesByType(['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca'],
                                               ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'],
                                               self.target_columns)

    def test_read_data(self):
        self.assertTupleEqual((297, 14), self.fake_data.shape)
        self.assertTrue(self.target_columns[0] in self.fake_data.columns)

    def test_split_labels_data(self):
        features, labels = split_features_labels_data(self.fake_data, self.features_by_type)

        self.assertTrue(self.target_columns[0] not in features.columns)
        self.assertEqual(1, labels.shape[1])

    def test_split_train_test_data(self):
        test_size, random_state = 0.2, 42
        splitting_params = SplittingParams(test_size,
                                           random_state)
        features, labels = split_features_labels_data(self.fake_data, self.features_by_type)
        x_train, x_test, y_train, y_test = split_train_test_data(features,
                                                                 labels,
                                                                 splitting_params)
        self.assertGreater(len(x_train), len(x_test) * 3)
        self.assertGreater(len(y_train), len(y_test) * 3)
        self.assertTrue(all(x_train.columns == x_test.columns))
