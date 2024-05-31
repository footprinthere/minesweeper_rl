from typing import Optional

import torch
from torch import Tensor, nn

from game import MineSweeperEnv


class MaxQValTracker:

    def __init__(
        self,
        env: MineSweeperEnv,
        policy_net: nn.Module,
        device: torch.device,
        use_mask: bool = False,
    ):
        self.env = env
        self.policy_net = policy_net
        self.device = device
        self.use_mask = use_mask

        self.samples: list[Tensor] = []  # [1, H, W]

    def collect_samples(self, size: int):
        """Collect sample states from the environment, using random actions."""

        state = None
        while len(self.samples) < size:
            if state is None:
                # Reset the environment
                self.env.reset()
                state = self.env.get_state()
            else:
                # Sample the next state
                state = self._sampling_step()

            if state is not None:
                self.samples.append(state.to(self.device))

    def get_max_q(self) -> tuple[float, float]:
        """Calculate the average Q-value of the collected samples."""

        max_q_sum = 0.0
        max_q_softmax_sum = 0.0
        for state in self.samples:
            max_q, max_q_softmax = self._compute_max_q(state)
            max_q_sum += max_q
            max_q_softmax_sum += max_q_softmax

        return max_q_sum / len(self.samples), max_q_softmax_sum / len(self.samples)

    def _sampling_step(self) -> Optional[Tensor]:
        """Perform one step of sampling."""

        action = self.env.sample_action(exclude_opened=True)
        env_step_result = self.env.step(action)
        if env_step_result.terminated:
            return None
        else:
            return torch.tensor(
                env_step_result.visible_board, dtype=torch.float32
            ).unsqueeze(0)

    def _compute_max_q(self, state: Tensor) -> tuple[float, float]:
        """Compute the Q-value of the given state."""

        with torch.no_grad():
            output = self.policy_net(state)
        if self.use_mask:
            mask = (state != self.env.CLOSED_CELL).flatten().to(self.device)
            output = torch.masked_fill(output, mask=mask, value=-1e9)
            # Cannot use env.get_open_mask() because the state is not from the environment

        max_q = torch.max(output).item()
        max_q_softmax = torch.max(torch.softmax(output, dim=1)).item()

        return max_q, max_q_softmax
