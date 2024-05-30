from dataclasses import dataclass


@dataclass
class TrainParameter:

    batch_size: int
    memory_size: int = 1000
    q_sample_size: int = 20

    gamma: float = 0.9
    eps_range: tuple[float, float] = (0.9, 0.005)
    eps_decay: int = 1000
    tau: float = 0.005
    lr: float = 0.005
    grad_max_norm: float = 1.0

    use_action_mask: bool = True
