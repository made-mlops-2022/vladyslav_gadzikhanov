# import os
#
# import click
# from sklearn.datasets import load_wine
#
#
# @click.command("download")
# @click.argument("output_dir")
# def download(output_dir: str):
#     X, y = load_wine(return_X_y=True, as_frame=True)
#
#     os.makedirs(output_dir, exist_ok=True)
#     X.to_csv(os.path.join(output_dir, "data.csv"))
#
#
# if __name__ == '__main__':
#     download()

import os
import random
import numpy as np
import pandas as pd
import argparse
from scipy.stats import gaussian_kde
from features import FeaturesByType


def data_generator(df: pd.DataFrame, features_by_type: FeaturesByType):
    output_data = []
    columns = []
    size, seed = df.shape[0], random.randint(0, 100)
    for feature in features_by_type.continuous_features:
        columns.append(feature)

        generated_col = gaussian_kde(df[feature]).resample(size=size, seed=seed)
        output_data.append(generated_col[0])

    categorical_and_target = features_by_type.categorical_features
    categorical_and_target.extend(features_by_type.target_columns)
    for feature in categorical_and_target:
        columns.append(feature)
        value_counts = df[feature].value_counts()
        values, weights = value_counts.index, value_counts.values

        generated_col = random.choices(population=values, weights=weights, k=size)
        output_data.append(generated_col)

    output_data = pd.DataFrame(np.array(output_data).T, columns=columns, )
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    original_data_name = "heart_cleveland_upload.csv"
    original_data = pd.read_csv(original_data_name)
    features_by_type = FeaturesByType(['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca'],
                                      ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'],
                                      ['condition'])
    fake_data = data_generator(original_data, features_by_type)

    os.makedirs(args.output_dir, exist_ok=True)
    fake_data.to_csv(os.path.join(args.output_dir, "data.csv"), index_label=False)
