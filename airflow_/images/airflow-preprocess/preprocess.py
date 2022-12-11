# import os
# import pandas as pd
# import click
#
#
# @click.command("predict")
# @click.option("--input-dir")
# @click.option("--output-dir")
# def preprocess(input_dir: str, output_dir):
#     data = pd.read_csv(os.path.join(input_dir, "data.csv"))
#     # do something instead
#     data["features"] = 0
#
#     os.makedirs(output_dir, exist_ok=True)
#     data.to_csv(os.path.join(output_dir, "data.csv"))
#
#
# if __name__ == '__main__':
#     preprocess()


import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from build_features import build_transformer, make_features
from features import FeaturesByType


def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    features, labels = data.iloc[:, :-1], data.iloc[:, -1]
    features_by_type = FeaturesByType(['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca'],
                                      ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'],
                                      ['condition'])
    preprocessed_features = make_features(build_transformer(features_by_type), features)

    os.makedirs(output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_features,
                                                        labels,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        random_state=42)

    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index_label=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index_label=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index_label=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index_label=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    preprocess(args.input_dir, args.output_dir)
