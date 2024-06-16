from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    name: str
    path_to_model_weight: str
    # activation: str