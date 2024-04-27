from typing import Optional

import torch
from torch import nn, optim, Tensor

from model import MineSweeperCNN
from game import MineSweeperEnv
from .memory import ReplayMemory
from .step_result import StepResult


class MineSweeperTrainer:

    def __init__(
        self,
        env: MineSweeperEnv,
        board_size: int,
        n_channels: int,
        model_depth: int,
        batch_size: int = 64,
        gamma: float = 0.99,
        eps_range: tuple[float, float] = (0.9, 0.05),
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
        memory_size: int = 1000,
    ):
        self.env = env

        self.policy_net = MineSweeperCNN(
            board_size=board_size, n_channels=n_channels, depth=model_depth
        )
        self.target_net = MineSweeperCNN(
            board_size=board_size, n_channels=n_channels, depth=model_depth
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_range
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.memory = ReplayMemory(capacity=memory_size)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0
        self.logs = {
            "loss": [],
            "duration": [],
            "max_q": [],
            "win": [],
        }

    def train(self, n_episodes: int, log_file: Optional[str] = None):
        pass

    def save_models(path: str):
        pass

    def plot_result(self, file_prefix: str):
        pass

    def step(self, state: Tensor) -> StepResult:
        pass

    def optimize(self) -> Optional[float]:
        pass

    def select_action(self, state: Tensor) -> Tensor:
        pass
