import argparse
import pickle
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression


def fit_model(input_dir: str, output_dir: str):
    model = LogisticRegression()

    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "fitted_model.pkl"), "wb") as output_model_file:
        pickle.dump(model, output_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    fit_model(args.input_dir, args.output_dir)
