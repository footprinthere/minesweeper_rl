from typing import Optional

import random
from collections import deque

from .open_result import OpenResult


class MineSweeperBoard:

    # Cells with mines (for hidden board)
    MINE = -1
    # Cells that are not open yet (for visible board)
    CLOSED = -2

    def __init__(self, height: int, width: int, n_mines: int):
        if height * width <= n_mines:
            raise ValueError("Number of mines must be less than the number of cells.")

        self.height = height
        self.width = width
        self.n_mines = n_mines

        # Initialized in reset_board()
        self._visible_board: list[list[int]]
        self._hidden_board: list[list[int]]
        self._open_state: list[list[bool]]
        self.n_closed: int

    @property
    def visible_board(self) -> list[list[int]]:
        return self._visible_board

    @property
    def open_state(self) -> list[list[bool]]:
        return self._open_state

    def __str__(self) -> str:
        return self._board_to_str(
            self._visible_board, colored=[MineSweeperBoard.CLOSED]
        )

    def reset_board(self, seed: Optional[int] = None) -> None:
        """Resets the board to a new game state."""

        random.seed(seed)

        # The hidden board that contains all information about the mines
        self._hidden_board = self._init_hidden_board()
        # The visible board that the player sees
        self._visible_board = [
            [MineSweeperBoard.CLOSED] * self.width for _ in range(self.height)
        ]
        # The open state of each cell
        self._open_state = [[False] * self.width for _ in range(self.height)]
        # Number of closed cells
        self.n_closed = self.width * self.height

    def open_cell(self, x: int, y: int) -> OpenResult:
        """Opens a cell and returns the result of the step."""

        if self.is_opened(x, y):
            return OpenResult.DUPLICATED
        if self._hidden_board[x][y] == MineSweeperBoard.MINE:
            return OpenResult.FAIL

        is_adjacent = self._is_adjacent(x, y)

        queue = deque([(x, y)])  # BFS queue
        while queue:
            qx, qy = queue.popleft()
            if self.is_opened(qx, qy):
                continue  # already opened

            self._visible_board[qx][qy] = self._hidden_board[qx][qy]
            self._open_state[qx][qy] = True
            self.n_closed -= 1

            # Open neighboring positions if 0 is found
            if self._hidden_board[qx][qy] > 0:
                continue
            for ax, ay in self._around(qx, qy):
                queue.append((ax, ay))

        if self.n_closed == self.n_mines:
            return OpenResult.WIN
        elif is_adjacent:
            return OpenResult.ADJACENT
        else:
            return OpenResult.ISOLATED

    def is_opened(self, x: int, y: int) -> bool:
        """Checks if a cell is opened."""

        return self._open_state[x][y]

    def _init_hidden_board(self) -> list[list[int]]:
        """Fills the hidden board with mines and numbers."""

        board = [[0] * self.width for _ in range(self.height)]

        # Randomly select the positions of the mines
        mine_positions = random.sample(range(self.width * self.height), self.n_mines)

        # Fill mines and numbers
        for pos in mine_positions:
            x = pos // self.width
            y = pos % self.width

            board[x][y] = MineSweeperBoard.MINE

            for ax, ay in self._around(x, y):
                if board[ax][ay] != MineSweeperBoard.MINE:
                    board[ax][ay] += 1

        return board

    def _around(self, x: int, y: int):
        """Generates the positions around a cell."""

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue

                if 0 <= x + dx < self.height and 0 <= y + dy < self.width:
                    yield x + dx, y + dy

    def _is_adjacent(self, x: int, y: int) -> bool:
        """Checks if a cell is adjacent to a opened cell."""

        for ax, ay in self._around(x, y):
            if self.is_opened(ax, ay):
                return True
        return False

    @staticmethod
    def _board_to_str(
        board: list[list[int]], colored: Optional[list[int]] = None
    ) -> str:
        """Converts the board to a string."""

        if colored is None:
            colored = []

        output = ""
        for row in board:
            for cell in row:
                if cell in colored:
                    output += f"\033[31m{cell: >3}\033[37m"
                else:
                    output += f"{cell: >3}"
            output += "\n"

        return output
