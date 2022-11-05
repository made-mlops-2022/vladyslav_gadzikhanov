from unittest import TestCase
from mlops_hw1.data.make_dataset import read_data, split_features_labels_data
from mlops_hw1.enties.train_params import TrainingParams
from mlops_hw1.enties.features import FeaturesByType
from mlops_hw1.models.model_fit_predict import (train_model,
                                                predict_model,
                                                evaluate_model)
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from mlops_hw1.features.build_features import build_transformer, make_features


class TestModelFitPredict(TestCase):
    random_state = 42
    output_model_path = './test_model.pkl'
    output_predictions_path = './test_predictions.csv'
    output_metrics_path = './metrics.json'

    def return_train_params(self, model_name):
        train_params = TrainingParams(model_name=model_name,
                                      random_state=self.random_state)
        return train_params

    def setUp(self):
        self.fake_data = read_data('../../data/fake_data.csv')
        self.features_by_type = FeaturesByType(['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca'],
                                               ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'],
                                               ['condition'])
        self.features, self.labels = split_features_labels_data(self.fake_data,
                                                                features_by_type=self.features_by_type)
        self.updated_features = make_features(build_transformer(self.features_by_type), self.features)

    def test_train_model_right(self):
        correct_models = [("CatBoostClassifier", CatBoostClassifier),
                          ("LogisticRegression", LogisticRegression)]

        for model_name, model_type in correct_models:
            train_params = self.return_train_params(model_name)

            model = train_model(self.updated_features, self.labels,  train_params, self.output_model_path)
            self.assertTrue(isinstance(model, model_type))

    def test_train_model_wrong(self):
        incorrect_model_name, incorrect_model_type = "LinearRegression", LinearRegression
        train_params = self.return_train_params(incorrect_model_name)

        with self.assertRaises(NotImplementedError):
            train_model(self.updated_features, self.labels, train_params, self.output_model_path)

    def test_predict_and_evaluate_model(self):
        predicted_labels, predicted_probas = predict_model(x_test=self.updated_features,
                                                           input_model_path=self.output_model_path,
                                                           output_predictions_path=self.output_predictions_path)

        self.assertEqual(len(self.updated_features), len(predicted_labels))
        self.assertEqual(len(self.updated_features), len(predicted_probas))
        self.assertSetEqual(set([0, 1]), set(predicted_labels))

        metrics = evaluate_model(predicted_labels=predicted_labels,
                                 predicted_probas=predicted_probas,
                                 labels=self.labels,
                                 output_metrics_path=self.output_metrics_path)

        for metric in ["accuracy", "f1", "roc_auc"]:
            self.assertGreater(metrics[metric], 0)
            self.assertLess(metrics[metric], 1)
