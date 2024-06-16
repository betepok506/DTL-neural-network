from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass, field

@dataclass()
class TrainingParams:
    lr: float
    num_train_epochs: int
    use_augmentation: bool
    freeze_layers: bool
    # is_clip_grad_norm: bool
    # is_clip_grad_value: bool

    # criterion: CriterionParams
    # optimizer: OptimizerParams
    # scheduler: SchedulerParams

    image_size: list
    image_crop: list
    train_batch_size: int
    test_batch_size: int
    # verbose: int
    output_dir_result: str
    save_to_checkpoint: str
    num_workers_data_loader: int
    report_to: str