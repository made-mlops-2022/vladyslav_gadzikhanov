# import os
# import pandas as pd
#
# import click
#
#
# @click.command("predict")
# @click.option("--input-dir")
# @click.option("--output-dir")
# def predict(input_dir: str, output_dir):
#     data = pd.read_csv(os.path.join(input_dir, "data.csv"))
#     # do real predict instead
#     data["predict"] = 1
#
#     os.makedirs(output_dir, exist_ok=True)
#     data.to_csv(os.path.join(output_dir, "data.csv"))
#
#
# if __name__ == '__main__':
#     predict()

import os
import pickle
import pandas as pd
import argparse


def predict(model_dir: str, data_dir: str, output_dir: str):
    with open(os.path.join(model_dir, "fitted_model.pkl"), "rb") as model_file:
        model = pickle.load(model_file)

    X_train = pd.read_csv(os.path.join(data_dir, "X_test.csv"))

    predictions = model.predict(X_train)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(predictions).to_csv(os.path.join(output_dir, "predictions.csv"), index_label=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    predict(args.model_dir, args.data_dir, args.output_dir)
