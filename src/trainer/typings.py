from typing import TypedDict


class TrainerConfig(TypedDict):
    class DataConfig(TypedDict):
      uri: str
      filename: str
      attributes: [str] or [int]

    class TrainConfig(TypedDict):
        train_size: float
        output_path: str
        fit_params: dict

    data: DataConfig
    train: TrainConfig
    random_seed: int