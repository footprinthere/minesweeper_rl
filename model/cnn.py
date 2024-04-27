from torch import nn, Tensor
import torch.nn.functional as F


class MineSweeperCNN(nn.Module):

    def __init__(
        self,
        board_size: int,
        n_channels: int,
        depth: int,
        padding: int | str = "same",
    ):
        super().__init__()

        self._input = nn.Conv2d(1, n_channels, (3, 3), padding=padding)
        self._cnns = nn.ModuleList(
            [
                nn.Conv2d(n_channels, n_channels, (3, 3), padding=padding)
                for _ in range(depth)
            ]
        )
        self._ff = nn.Linear(
            in_features=n_channels * board_size, out_features=board_size
        )

    def forward(self, state: Tensor) -> Tensor:
        # state: (N, H, W)
        x = state.unsqueeze(1)  # (N, 1, H, W)
        x = F.relu(self._input(x))  # (N, C, H, W)
        for layer in self._cnns:
            x = F.relu(layer(x))

        x = x.reshape(x.size(0), -1)  # (N, C*H*W)
        x = self._ff(x)
        return x
