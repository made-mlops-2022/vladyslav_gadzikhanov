import json
import pickle
from typing import NoReturn


def save_metrics(path: str,
                 metrics: dict) -> NoReturn:
    with open(path, "w") as metrics_file:
        json.dump(metrics, metrics_file)


def save_model(path: str,
               model) -> NoReturn:
    with open(path, "wb") as output_model_file:
        pickle.dump(model, output_model_file)
