from dataclasses import dataclass


@dataclass
class ModelParameter:

    board_size: int
    n_channels: int
    cnn_depth: int
    ff_dim: int
