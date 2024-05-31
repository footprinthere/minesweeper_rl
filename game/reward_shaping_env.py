from overrides import overrides

from torch import Tensor

from game.env import MineSweeperEnv
from game.env_step_result import EnvStepResult
from game.open_result import OpenResult


class MineSweeperRewardShapingEnv(MineSweeperEnv):

    def __init__(
        self,
        board_height: int,
        board_width: int,
        n_mines: int,
        gamma: float,
    ):
        super().__init__(board_height, board_width, n_mines)

        self.gamma = gamma

    @overrides
    def step(self, action: int) -> EnvStepResult:
        prev_state = self.get_state()
        assert prev_state is not None, "Env not initialized"

        x, y = action // self._board.width, action % self._board.width
        open_result = self._board.open_cell(x, y)

        terminated = open_result in {OpenResult.FAIL, OpenResult.WIN}
        if not terminated:
            self._curr_state = MineSweeperEnv._board_to_tensor(
                self._board.visible_board
            )
        else:
            self._curr_state = None

        if open_result == OpenResult.ADJACENT:
            curr_state = self.get_state()
            assert curr_state is not None, "Game is terminated"
            reward = (
                1 + self._get_shaped_reward(prev_state, curr_state)
            ) * MineSweeperEnv._reward_map[OpenResult.ADJACENT]
        else:
            reward = MineSweeperEnv._reward_map[open_result]

        return EnvStepResult(
            visible_board=self._board.visible_board,
            reward=reward,
            terminated=terminated,
            open_result=open_result,
        )

    def _get_shaped_reward(self, prev_state: Tensor, curr_state: Tensor) -> float:
        return self.gamma * self._potential(curr_state) - self._potential(prev_state)

    def _potential(self, state: Tensor) -> float:
        """Calculates ratio of opened cells"""
        compared = state == self.CLOSED_CELL
        n_closed = compared.sum().item()

        return 1 - (n_closed / compared.numel())
