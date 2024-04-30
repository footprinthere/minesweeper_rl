from dataclasses import dataclass


@dataclass
class ModelParameter:

    board_size: int
    n_channels: int
    cnn_depth: int
    ff_dim: int


@dataclass
class TrainParameter:

    batch_size: int
    memory_size: int = 1000
    q_sample_size: int = 20

    gamma: float = 0.99
    eps_range: tuple[float, float] = (0.9, 0.05)
    eps_decay: int = 1000
    tau: float = 0.005
    lr: float = 1e-4
    grad_max_norm: float = 1.0

    use_action_mask: bool = True
