import pandas as pd
import requests


PATH_TO_DATASET = "heart_cleveland_upload.csv"
TARGET_COLUMN = "condition"
ENDPOINT_URL = "http://0.0.0.0:8000/predict"
CATEGORICAL_FEATURES = ["sex", "fbs", "restecg", "exang", "slope", "thal", "cp", "ca"]
CONTINUOUS_FEATURES = ["trestbps", "chol", "thalach", "oldpeak", "age"]


if __name__ == "__main__":
    data = pd.read_csv(PATH_TO_DATASET).drop(columns=TARGET_COLUMN)

    requests_params = {"data": data.to_json(),
                       "continuous_features": CONTINUOUS_FEATURES,
                       "categorical_features": CATEGORICAL_FEATURES}

    response = requests.post(ENDPOINT_URL, json=requests_params)
    print(response.status_code)
    print(response.json())
