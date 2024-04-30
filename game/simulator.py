from typing import Optional

from .env import MineSweeperEnv


class MineSweeperSimulator:
    def __init__(
        self,
        height: int,
        width: int,
        n_mines: int,
        seed: Optional[int] = None,
    ):
        self.env = MineSweeperEnv(height, width, n_mines)
        self.env.reset(seed=seed)

        self.pos_to_action = lambda x, y: x * width + y

    def play(self) -> None:
        while True:
            print(self.env.render())
            inp = input("(x, y) >>> ").split()
            x, y = int(inp[0]), int(inp[1])

            result = self.env.step(self.pos_to_action(x, y))
            print(result.open_result, f"/ Reward: {result.reward}")
            print()

            if result.terminated:
                break
