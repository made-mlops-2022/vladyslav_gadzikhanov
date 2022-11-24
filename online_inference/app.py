import os
import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from typing import NoReturn, List
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
# from mlops_hw2.hw1_files.features import FeaturesByType
# from mlops_hw2.hw1_files.build_features import build_transformer, make_features
from features import FeaturesByType
from build_features import build_transformer, make_features


app_ = FastAPI()
model = None


class HeartConditionDataset(BaseModel):
    data: str
    continuous_features: List[str]
    categorical_features: List[str]

    class Config:
        arbitrary_types_allowed = True


class HeartConditionPrediction(BaseModel):
    condition: int


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as file:
        return pickle.load(file)


def preprocess_data(
        data: pd.DataFrame, categorical_features: list, continuous_features: list
) -> pd.DataFrame:
    features_by_types = FeaturesByType(categorical_features=categorical_features,
                                       continuous_features=continuous_features,
                                       target_columns=list())
    transformer = build_transformer(features_by_types)
    updated_data = make_features(transformer, data)
    return updated_data


def make_predict(
        data: pd.DataFrame, model
) -> list[HeartConditionPrediction]:

    predictions = model.predict(data)

    return [
        HeartConditionPrediction(condition=item) for item in predictions
    ]


@app_.get("/")
async def main():
    return "it's entry point of our predictor"


@app_.on_event("startup")
async def load_model() -> NoReturn:
    global model
    # os.environ["PATH_TO_MODEL"] = "models/LogisticRegression_model.pkl"
    path_to_model = os.getenv("PATH_TO_MODEL")
    if path_to_model is None:
        raise RuntimeError()

    model = load_object(path_to_model)


@app_.get("/health")
async def health() -> int:
    return 200 if model is not None else 404


@app_.post("/predict")
async def predict(request: HeartConditionDataset):
    global model
    preprocessed_data = preprocess_data(data=pd.read_json(request.data),
                                        continuous_features=request.continuous_features,
                                        categorical_features=request.categorical_features)
    return make_predict(data=preprocessed_data,
                        model=model)


if __name__ == "__main__":
    uvicorn.run(app=app_,
                host="0.0.0.0",
                port=8000)
