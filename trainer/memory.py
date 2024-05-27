from typing import Optional
from collections import deque
from dataclasses import dataclass
import random

import torch
from torch import Tensor


@dataclass
class Transition:
    state: Tensor  # [1, H, W]
    action: Tensor  # [1,]
    reward: Tensor  # [1,]
    next_state: Optional[Tensor]  # `None` if episode terminated

    @staticmethod
    def concat(transitions: list["Transition"]) -> tuple["Transition", Tensor]:
        states, actions, next_states, rewards = [], [], [], []
        for t in transitions:
            states.append(t.state)
            actions.append(t.action)
            next_states.append(t.next_state)
            rewards.append(t.reward)

        # `next_states` may contain `None` if episode terminated
        non_final_mask = torch.tensor(
            [s is not None for s in next_states], dtype=torch.bool
        )
        non_final_next_states = [s for s in next_states if s is not None]

        return (
            Transition(
                state=torch.cat(states),
                action=torch.cat(actions),
                reward=torch.cat(rewards),
                next_state=torch.cat(non_final_next_states),
            ),
            non_final_mask,
        )

    def to_device(self, device: torch.device) -> None:
        self.state = self.state.to(device)
        self.action = self.action.to(device)
        self.reward = self.reward.to(device)
        if self.next_state is not None:
            self.next_state = self.next_state.to(device)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    @property
    def size(self) -> int:
        return len(self.memory)

    def push(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Optional[Tensor],
    ) -> None:
        self.memory.append(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, k=batch_size)
