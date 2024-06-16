from dataclasses import dataclass, field


@dataclass()
class DatasetParams:
    path_to_train_data: str
    path_to_test_data: str
    # path_to_decode_classes2rgb: str
    # ignore_index: int
    num_labels: int