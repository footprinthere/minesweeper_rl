from typing import Optional
import os

import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from game import MineSweeperEnv
from model import MineSweeperCNN, ModelParameter


class MineSweeperPlayer:

    def __init__(
        self,
        env: MineSweeperEnv,
        model_param: ModelParameter,
        project_dir: str,
        device: torch.device,
    ):
        self.env = env

        self.policy_net = MineSweeperCNN.with_parameter(model_param)
        self.target_net = MineSweeperCNN.with_parameter(model_param)
        self.project_dir = project_dir
        self.load_models(self.project_dir)

        self.device = device
        self.policy_net.eval().to(self.device)
        self.target_net.eval().to(self.device)

        self.rewards: list[float] = []

    def load_models(self, directory: str) -> None:
        path_format = os.path.join(directory, "{}.pt")
        self.policy_net.load_state(path_format.format("policy"))
        self.target_net.load_state(path_format.format("target"))

    def play(self, output_file: Optional[str] = None) -> None:

        def _write(content: str):
            if output_file is None:
                return
            if output_file == "stdin":
                tqdm.write(content)
                return

            with open(os.path.join(self.project_dir, output_file), "a") as f:
                f.write(content + "\n")

        self.env.reset()

        while True:
            self._step()
            _write(self.env.render())
            if self.env.is_terminated():
                break

        self.plot_logs()

    def _step(self) -> None:
        prev_state = self.env.get_state()
        assert prev_state is not None
        action = self._select_action(use_mask=True)

        env_step_result = self.env.step(action)
        self.rewards.append(env_step_result.reward)

    def _select_action(self, use_mask: bool) -> int:
        output = self._compute_q(use_mask=use_mask)
        argmax = torch.max(output, dim=1).indices
        return int(argmax.item())

    def _compute_q(self, use_mask: bool) -> Tensor:
        """Computes Q(s, a...) with the policy network"""

        if (state := self.env.get_state()) is None:
            raise ValueError("Episode is already terminated")
        with torch.no_grad():
            output = self.policy_net(state.to(self.device))
        if use_mask:
            output = torch.masked_fill(
                output, mask=self.env.get_open_mask().to(self.device), value=-1e9
            )

        return output

    def plot_logs(self) -> None:
        plt.title("REWARD")
        plt.plot(self.rewards)
        plt.show()
