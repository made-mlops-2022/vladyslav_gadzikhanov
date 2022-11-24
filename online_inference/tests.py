import pandas as pd
# from mlops_hw2.app import app_
from app import app_
from unittest import TestCase
from fastapi.testclient import TestClient


class TestApp(TestCase):
    TARGET_COLUMN = "condition"
    PATH_TO_DATASET = "heart_cleveland_upload.csv"
    CATEGORICAL_FEATURES = ["sex", "fbs", "restecg", "exang", "slope", "thal", "cp", "ca"]
    CONTINUOUS_FEATURES = ["trestbps", "chol", "thalach", "oldpeak", "age"]

    def test_read_main(self):
        with TestClient(app_) as client:
            response = client.get("/")
            self.assertEqual(200, response.status_code)
            self.assertEqual("it's entry point of our predictor", response.json())

    def test_predict(self):
        data = pd.read_csv(self.PATH_TO_DATASET).drop(columns=self.TARGET_COLUMN)

        requests_params = {"data": data.to_json(),
                           "continuous_features": self.CONTINUOUS_FEATURES,
                           "categorical_features": self.CATEGORICAL_FEATURES}
        with TestClient(app_) as client:
            response = client.post("/predict", json=requests_params)
            self.assertEqual(200, response.status_code)
            self.assertEqual(data.shape[0], len(response.json()))

            for item in response.json():
                self.assertListEqual([self.TARGET_COLUMN], list(item.keys()))
                self.assertIn(member=item[self.TARGET_COLUMN],
                              container=[0, 1])



