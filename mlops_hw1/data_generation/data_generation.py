import random
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from mlops_hw1.enties.features import FeaturesByType


def data_generator(df: pd.DataFrame, features_by_type: FeaturesByType):
    output_data = []
    columns = []
    size, seed = df.shape[0], 42
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


def main():
    original_data = pd.read_csv('../data/heart_cleveland_upload.csv')
    features_by_type = FeaturesByType(['sex', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'cp', 'ca'],
                                      ['trestbps', 'chol', 'thalach', 'oldpeak', 'age'],
                                      ['condition'])
    fake_data = data_generator(original_data, features_by_type)
    fake_data.to_csv('../data/fake_data.csv', index_label=False)


if __name__ == "__main__":
    main()
