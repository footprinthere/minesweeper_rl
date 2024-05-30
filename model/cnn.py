import torch
from torch import nn, Tensor
import torch.nn.functional as F

from model.parameter import ModelParameter


class MineSweeperCNN(nn.Module):

    def __init__(
        self,
        board_size: int,
        n_channels: int,
        cnn_depth: int,
        ff_dim: int,
        padding: int | str = "same",
    ):
        super().__init__()

        self._input = nn.Conv2d(1, n_channels, (3, 3), padding=padding)
        self._cnns = nn.ModuleList(
            [
                nn.Conv2d(n_channels, n_channels, (3, 3), padding=padding)
                for _ in range(cnn_depth)
            ]
        )
        self._ffs = nn.ModuleList(
            [
                nn.Linear(in_features=n_channels * board_size, out_features=ff_dim),
                nn.Linear(in_features=ff_dim, out_features=ff_dim),
            ]
        )
        self._output = nn.Linear(in_features=ff_dim, out_features=board_size)

    @classmethod
    def with_parameter(cls, parameter: ModelParameter) -> "MineSweeperCNN":
        return cls(
            board_size=parameter.board_size,
            n_channels=parameter.n_channels,
            cnn_depth=parameter.cnn_depth,
            ff_dim=parameter.ff_dim,
        )

    def forward(self, state: Tensor) -> Tensor:
        # state: (N, H, W)
        x = state.unsqueeze(1)  # (N, 1, H, W)
        x = F.relu(self._input(x))  # (N, C, H, W)
        for layer in self._cnns:
            x = F.relu(layer(x))

        x = torch.flatten(x, start_dim=1)  # (N, C*H*W)
        for layer in self._ffs:
            x = F.relu(layer(x))
        x = self._output(x)  # (N, H*W)
        return x

    def save_state(self, path: str) -> None:
        if not path.endswith(".pt"):
            raise ValueError("Model path must end with '.pt'")
        torch.save(self.state_dict(), path)

    def load_state(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
