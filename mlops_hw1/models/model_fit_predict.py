import pickle
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from mlops_hw1.data.make_dataset import read_data
from mlops_hw1.save_results.save_results import save_metrics, save_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from mlops_hw1.features.build_features import build_transformer, make_features
from mlops_hw1.models.defining_logger_configuration import get_logger
from mlops_hw1.data.make_dataset import split_features_labels_data, split_train_test_data
from mlops_hw1.enties.train_params import TrainingParams
from mlops_hw1.enties.train_pipline_params import read_training_pipeline_params


def train_model(x_train: pd.DataFrame,
                y_train: pd.DataFrame,
                train_params: TrainingParams,
                output_model_path: str):
    if train_params.model_name == "CatBoostClassifier":
        model = CatBoostClassifier(n_estimators=300,
                                   learning_rate=0.2,
                                   random_seed=train_params.random_state,
                                   depth=3,
                                   verbose=False
                                   )
    elif train_params.model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000,
                                   random_state=train_params.random_state
                                   )
    else:
        raise NotImplementedError("Unidentified model found")

    model.fit(x_train, y_train)
    save_model(path=output_model_path,
               model=model)

    return model


def predict_model(x_test, output_predictions_path, input_model_path):
    with open(input_model_path, 'rb') as file:
        model = pickle.load(file)

    predicted_labels = model.predict(x_test)
    predicted_probas = model.predict_proba(x_test)[:, 1]

    columns = ['predicted_labels', 'predicted_probas']
    output = np.array([predicted_labels, predicted_labels])

    pd.DataFrame(output.T, columns=columns).to_csv(output_predictions_path, index_label=False)

    return predicted_labels, predicted_probas


def evaluate_model(predicted_labels: np.ndarray,
                   predicted_probas: np.ndarray,
                   labels: pd.DataFrame,
                   output_metrics_path
                   ):
    metrics = {
        "accuracy": accuracy_score(labels, predicted_labels),
        "f1": f1_score(labels, predicted_labels),
        "roc_auc": roc_auc_score(labels, predicted_probas)
    }
    save_metrics(output_metrics_path, metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('flag', type=str)
    args = parser.parse_args()
    logger = get_logger()
    training_pipeline_params = read_training_pipeline_params(args.config_path)

    logger.info(f'Start read_data with params: {training_pipeline_params.input_data_path}')
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f'data params: {type(data)=}, {data.shape=}\n')

    logger.info(f'\nStart split_obj_labels_data with params:'
                f'\tdata: {type(data)=}, {data.shape=}\n'
                f'\tvariables_by_type={training_pipeline_params.features_by_type}')
    features, labels = split_features_labels_data(data, training_pipeline_params.features_by_type)
    logger.info(f'\nfeatures params: {type(features)=}, {features.shape=}\n'
                f'labels params: {type(labels)=}, {labels.shape=}\n')

    transformer = build_transformer(training_pipeline_params.features_by_type)
    updated_features = make_features(transformer, features)

    logger.info(f'\nStart split_train_test_data with params:\n'
                f'\tfeatures: {type(features)=}, {features.shape=}\n'
                f'\tlabels: {type(labels)=}, {labels.shape=}\n'
                f'\tsplitting_params: {training_pipeline_params.splitting_params}')
    x_train, x_test, y_train, y_test = split_train_test_data(features=updated_features,
                                                             labels=labels,
                                                             splitting_params=training_pipeline_params.splitting_params)
    logger.info(f'\nx_train params: {type(x_train)=}, {x_train.shape=}\n'
                f'x_test params: {type(x_test)=}, {x_test.shape=}\n'
                f'y_train params: {type(y_train)}, {y_train.shape=}\n'
                f'y_test params: {type(y_test)}, {y_test.shape=}\n')

    output_metrics_path = training_pipeline_params.output_metrics_path
    output_predictions_path = training_pipeline_params.output_predictions_path
    output_model_path = training_pipeline_params.output_model_path

    if args.flag == 'train':
        train_params = training_pipeline_params.train_params
        logger.info('\nStart train_model with params:\n'
                    f'\tx_train: {type(x_train)=}, {x_train.shape=}\n'
                    f'\ty_train: {type(y_train)=}, {y_train.shape=}\n'
                    f'\ttrain_params: {training_pipeline_params.train_params}\n'
                    f'\toutput_model_path: output_model_path')
        train_model(x_train, y_train, train_params, output_model_path)
        logger.info('train_model completed\n')
    elif args.flag == 'predict':
        logger.info(f'\nStart predict_model with params:\n'
                    f'\tx_test: {type(x_test)=}, {x_test.shape=}\n'
                    f'\toutput_predictions_path: {output_predictions_path}\n'
                    f'\toutput_model_path: {output_model_path}')
        predicted_labels, predicted_probas = predict_model(x_test, output_predictions_path, output_model_path)
        logger.info(f'\npredicted_labels params: {type(predicted_labels)=}, {predicted_labels.shape=}\n'
                    f'predicted_probas params: {type(predicted_probas)=}, {predicted_probas.shape=}\n')

        logger.info(f'\nStart evaluate_model with params:\n'
                    f'\tpredicted_labels params: {type(predicted_labels)=}, {predicted_labels.shape=}\n'
                    f'\tpredicted_probas params: {type(predicted_probas)=}, {predicted_probas.shape=}\n'
                    f'\tlabels: {type(y_test)=}, {y_test.shape=}\n'
                    f'\toutput_metrics_path: {output_metrics_path}')
        evaluate_model(predicted_labels, predicted_probas, y_test, output_metrics_path)
        logger.info('evaluate_model completed\n')


if __name__ == "__main__":
    main()
