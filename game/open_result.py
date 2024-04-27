from enum import Enum, auto


class OpenResult(int, Enum):
    """Enum representing the result of opening a cell in the MineSweeper game."""

    FAIL = auto()
    WIN = auto()
    ADJACENT = auto()
    ISOLATED = auto()
    DUPLICATED = auto()
