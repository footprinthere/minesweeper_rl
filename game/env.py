from typing import Optional
import random

from .board import MineSweeperBoard
from .open_result import OpenResult
from .env_step_result import EnvStepResult


class MineSweeperEnv:

    def __init__(self, board_height: int, board_width: int, n_mines: int):
        self._board = MineSweeperBoard(
            height=board_height, width=board_width, n_mines=n_mines
        )

        self._reward_map = {
            OpenResult.FAIL: -10,
            OpenResult.WIN: 10,
            OpenResult.ADJACENT: 1,
            OpenResult.ISOLATED: -1,
            OpenResult.DUPLICATED: -3,
        }

    def reset(self, seed: Optional[int] = None) -> list[list[int]]:
        """Resets the environment to a new game state. Should be called before `step()`."""

        self._board.reset_board(seed=seed)
        return self._board.visible_board

    def step(self, action: int) -> EnvStepResult:
        x, y = action // self._board.width, action % self._board.width
        result = self._board.open_cell(x, y)

        return EnvStepResult(
            visible_board=self._board.visible_board,
            reward=self._reward_map[result],
            terminated=result in {OpenResult.FAIL, OpenResult.WIN},
            open_result=result,
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

    def get_open_state(self) -> list[list[bool]]:
        return self._board.open_state

    def render(self) -> str:
        """Returns a string representation of the current game state."""

        return str(self._board)
