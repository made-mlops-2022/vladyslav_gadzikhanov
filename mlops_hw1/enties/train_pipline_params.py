from dataclasses import dataclass
from mlops_hw1.enties.split_params import SplittingParams
from mlops_hw1.enties.train_params import TrainingParams
from mlops_hw1.enties.features import FeaturesByType
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_metrics_path: str
    output_predictions_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    features_by_type: FeaturesByType


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
