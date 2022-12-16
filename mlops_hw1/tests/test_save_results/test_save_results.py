import json
import os
import pickle
from catboost import CatBoostClassifier
from mlops_hw1.save_results.save_results import save_metrics, save_model
from unittest import TestCase


class TestSaveResults(TestCase):
    @staticmethod
    def remove(path):
        os.remove(path)

    def test_save_metrics(self):
        path = "./test_metrics.json"
        metrics = {"accuracy": 0.95, "precision": 0.88}
        save_metrics(path, metrics)

        with open(path, "r") as json_file:
            read_metrics = json.load(json_file)

        self.assertDictEqual(metrics, read_metrics)
        self.remove(path)

    def test_save_model(self):
        path = "./test_model.pkl"
        model = CatBoostClassifier(n_estimators=1000,
                                   learning_rate=0.1,
                                   subsample=0.66,
                                   depth=3)
        save_model(path, model)

        with open(path, "rb") as pkl_file:
            read_model = pickle.load(pkl_file)

        self.assertTrue(isinstance(read_model, CatBoostClassifier))
        self.assertDictEqual(model.get_params(), read_model.get_params())
        self.remove(path)
