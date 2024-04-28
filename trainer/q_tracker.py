from typing import Optional

import torch
from torch import Tensor, nn

from game import MineSweeperEnv


class MaxQValTracker:
    def __init__(self, env: MineSweeperEnv):
        self.env = env
        self.samples: list[Tensor] = []  # [1, H, W]

    def collect_samples(self, size: int):
        """Collect sample states from the environment, using random actions."""

        state = None
        while len(self.samples) < size:
            if state is None:
                # Reset the environment
                state = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0)
            else:
                # Sample the next state
                state = self._sampling_step()

            if state is not None:
                self.samples.append(state)

    def get_max_q(self, policy_net: nn.Module) -> tuple[float, float]:
        """Calculate the average Q-value of the collected samples."""

        max_q_sum = 0.0
        max_q_softmax_sum = 0.0
        for state in self.samples:
            max_q, max_q_softmax = self._compute_max_q(policy_net, state)
            max_q_sum += max_q
            max_q_softmax_sum += max_q_softmax

        return max_q_sum / len(self.samples), max_q_softmax_sum / len(self.samples)

    def _sampling_step(self) -> Optional[Tensor]:
        """Perform one step of sampling."""

        action = self.env.sample_action()
        env_step_result = self.env.step(action)
        if env_step_result.terminated:
            return None
        else:
            return torch.tensor(
                env_step_result.visible_board, dtype=torch.float32
            ).unsqueeze(0)

    def _compute_max_q(
        self,
        policy_net: nn.Module,
        state: Tensor,
    ) -> tuple[float, float]:
        """Compute the Q-value of the given state."""

        with torch.no_grad():
            output = policy_net(state)

        max_q = torch.max(output).item()
        max_q_softmax = torch.max(torch.softmax(output, dim=1)).item()

        return max_q, max_q_softmax
