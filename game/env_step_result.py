from dataclasses import dataclass

from game.open_result import OpenResult


@dataclass
class EnvStepResult:
    """The result of a step in the game environment."""

    visible_board: list[list[int]]
    reward: int
    terminated: bool
    open_result: OpenResult
