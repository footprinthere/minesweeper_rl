from typing import Optional

import torch

from .board import MineSweeperBoard
from .open_result import OpenResult
from .env_step_result import EnvStepResult


class MineSweeperEnv:

    def __init__(self, board_height: int, board_width: int, n_mines: int):
        self.board = MineSweeperBoard(
            height=board_height, width=board_width, n_mines=n_mines
        )

        self.reward_map = {
            OpenResult.FAIL: -10,
            OpenResult.WIN: 10,
            OpenResult.ADJACENT: 1,
            OpenResult.ISOLATED: -1,
            OpenResult.DUPLICATED: -3,
        }

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Resets the environment to a new game state. Should be called before `step()`."""

        self.board.reset_board(seed=seed)
        return torch.tensor(self.board.visible_board, dtype=torch.float32)

    def step(self, action: int) -> EnvStepResult:
        x, y = action // self.board.width, action % self.board.width
        result = self.board.open(x, y)

        return EnvStepResult(
            visible_board=self.board.visible_board,
            reward=self.reward_map[result],
            terminated=result in {OpenResult.FAIL, OpenResult.WIN},
            open_result=result,
        )

    def sample_action(self) -> torch.Tensor:
        """Returns a random action."""

        return torch.randint(0, self.board.height * self.board.width, size=(1,))

    def render(self) -> str:
        """Returns a string representation of the current game state."""

        return str(self.board)
