import argparse
import json
import os

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def validation(right_labels_pass: str, predictions_path: str, output_path: str):
    right_labels = pd.read_csv(os.path.join(right_labels_pass, "y_test.csv"))
    predictions = pd.read_csv(os.path.join(predictions_path, "predictions.csv"))

    metrics = dict()
    metrics["f1_score"] = f1_score(right_labels.to_numpy(), predictions.to_numpy())
    metrics["accuracy_score"] = accuracy_score(right_labels.to_numpy(), predictions.to_numpy())

    # os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("right_labels_pass", type=str)
    parser.add_argument("predictions_pass", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    validation(args.right_labels_pass, args.predictions_pass, args.output_path)
