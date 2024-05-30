from typing import Optional
import random

import torch
from torch import Tensor

from .board import MineSweeperBoard
from .open_result import OpenResult
from .env_step_result import EnvStepResult


class MineSweeperEnv:

    _reward_map = {
        OpenResult.FAIL: -1,
        OpenResult.WIN: 1,
        OpenResult.ADJACENT: 0.3,
        OpenResult.ISOLATED: -0.3,
        OpenResult.DUPLICATED: -0.3,
    }

    def __init__(self, board_height: int, board_width: int, n_mines: int):
        self.board_height = board_height
        self.board_width = board_width
        self.n_mines = n_mines

        self._board = MineSweeperBoard(
            height=board_height, width=board_width, n_mines=n_mines
        )
        self._curr_state: Optional[Tensor] = None

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets the environment to a new game state. Should be called before `step()`."""

        self._board.reset_board(seed=seed)
        self._curr_state = MineSweeperEnv._board_to_tensor(self._board.visible_board)

    def step(self, action: int) -> EnvStepResult:
        x, y = action // self._board.width, action % self._board.width
        open_result = self._board.open_cell(x, y)

        terminated = open_result in {OpenResult.FAIL, OpenResult.WIN}
        if not terminated:
            self._curr_state = MineSweeperEnv._board_to_tensor(
                self._board.visible_board
            )
        else:
            self._curr_state = None

        return EnvStepResult(
            visible_board=self._board.visible_board,
            reward=MineSweeperEnv._reward_map[open_result],
            terminated=terminated,
            open_result=open_result,
        )

    def sample_action(self, exclude_opened: bool = False) -> int:
        """Returns a random action."""

        if not exclude_opened:
            return random.randrange(self._board.height * self._board.width)

        closed_cells = []
        for x in range(self._board.height):
            for y in range(self._board.width):
                if not self._board.is_opened(x, y):
                    closed_cells.append(x * self._board.width + y)

        return random.choice(closed_cells)

    def get_state(self) -> Optional[Tensor]:
        return self._curr_state

    def is_terminated(self) -> bool:
        return self._curr_state is None

    def get_board(self) -> list[list[int]]:
        return self._board.visible_board

    def get_open_state(self) -> list[list[bool]]:
        return self._board.open_state

    def get_open_mask(self) -> Tensor:
        # [H * W]
        return torch.tensor(self._board.open_state, dtype=torch.bool).flatten()

    def render(self) -> str:
        """Returns a string representation of the current game state."""

        return str(self._board)

    @staticmethod
    def _board_to_tensor(board: list[list[int]]) -> Tensor:
        # [1, H, W]
        return torch.tensor(board, dtype=torch.float32).unsqueeze(0)
