from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from dtl_siamese_network.enities.model_params import ModelParams
from dtl_siamese_network.enities.dataset_params import DatasetParams
from dtl_siamese_network.enities.training_params import TrainingParams
import yaml

@dataclass()
class TrainingConfig:
    comment: str
    short_comment: str

    # Model
    model: ModelParams

    # Dataset
    dataset: DatasetParams

    training_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingConfig)

def read_training_pipeline_params(path: str) -> TrainingConfig:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))