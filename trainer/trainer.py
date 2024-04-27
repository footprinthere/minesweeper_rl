from typing import Optional
from dataclasses import dataclass, field
from itertools import count
import math
import random

import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MineSweeperCNN
from game.env import MineSweeperEnv
from game.open_result import OpenResult
from .memory import ReplayMemory, Transition
from .train_step_result import TrainStepResult


class MineSweeperTrainer:

    def __init__(
        self,
        env: MineSweeperEnv,
        board_size: int,
        n_channels: int,
        model_depth: int,
        batch_size: int = 64,
        gamma: float = 0.99,
        eps_range: tuple[float, float] = (0.9, 0.05),
        eps_decay: int = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
        grad_max_norm: float = 1.0,
        memory_size: int = 1000,
    ):
        self.env = env

        self.policy_net = MineSweeperCNN(
            board_size=board_size, n_channels=n_channels, depth=model_depth
        )
        self.target_net = MineSweeperCNN(
            board_size=board_size, n_channels=n_channels, depth=model_depth
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_range
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.grad_max_norm = grad_max_norm
        self.memory = ReplayMemory(capacity=memory_size)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.steps_done = 0
        self.logs = TrainLog()

    def train(self, n_episodes: int) -> None:
        self.logs.clear()

        for i in tqdm(range(n_episodes)):
            # Reset environment
            state = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0)
            # [1, H, W]

            episode_loss = 0.0
            for t in count():
                step_result = self._step(state)
                if step_result.loss is not None:
                    episode_loss += step_result.loss

                if step_result.next_state is not None:
                    state = step_result.next_state
                else:
                    break  # episode terminated

            episode_loss /= t + 1
            self.logs.loss.append(episode_loss)
            self.logs.duration.append(t + 1)
            self.logs.win.append(step_result.open_result == OpenResult.WIN)

            tqdm.write(f"Episode {i} - loss {episode_loss :.4f}, duration {t + 1}")

    def save_models(self, path: str) -> None:
        raise NotImplementedError

    def plot_logs(self, log_dir: str) -> None:
        self.logs.plot(log_dir)

    def _step(self, state: Tensor) -> TrainStepResult:
        action = self._select_action(state)
        env_step_result = self.env.step(action)
        if env_step_result.terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                env_step_result.visible_board, dtype=torch.float32
            ).unsqueeze(0)

        # Store transition in memory
        # FIXME: tensor shape 검증 필요
        self.memory.push(
            state=state,
            action=torch.tensor([action]),
            reward=torch.tensor([env_step_result.reward]),
            next_state=next_state,
        )

        # Perform one step of optimization on the policy network
        self._optimize()

        # Soft update the target network's weights
        #   Alternatively, we can update the target network's weights every C steps
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        self.steps_done += 1
        return TrainStepResult(
            loss=None,
            next_state=next_state,
            open_result=env_step_result.open_result,
        )

    def _optimize(self) -> Optional[float]:
        if self.memory.size < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch, non_final_mask = Transition.concat(transitions)

        # Compute Q(s_t, a)
        #   The policy network returns Q(s),
        #   and then we choose the values corresponding to the given actions
        state_action_values = torch.gather(
            self.policy_net(batch.state), dim=1, index=batch.action
        )

        # Compute V(s_{t+1}) = R + γ * max_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(
                self.target_net(batch.next_state), dim=1
            ).values
        expected_state_action_values = next_state_values * self.gamma + batch.reward

        # Compute loss
        criterion = torch.nn.SmoothL1Loss()
        loss: Tensor = criterion(
            state_action_values, expected_state_action_values.unsqueeze(0)
        )

        # Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_max_norm)
        self.optimizer.step()

        return loss.item()

    def _select_action(self, state: Tensor, use_mask: bool = False) -> int:
        p = random.random()
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        if p < eps:
            # Explore
            return self.env.sample_action()

        # Exploit
        with torch.no_grad():
            output = self.policy_net(state)
        if use_mask:
            raise NotImplementedError

        _max = torch.max(output, dim=1)
        self.logs.max_q.append(_max.values.item())
        return int(_max.indices.item())


@dataclass
class TrainLog:
    loss: list[float] = field(default_factory=list)
    duration: list[int] = field(default_factory=list)
    win: list[bool] = field(default_factory=list)
    max_q: list[float] = field(default_factory=list)

    def clear(self):
        self.loss.clear()
        self.duration.clear()
        self.win.clear()
        self.max_q.clear()

    def plot(self, log_dir: str) -> None:
        # Loss
        plt.title("Loss")
        plt.xlabel("episodes")
        plt.ylabel("loss")
        plt.plot(self.loss)
        plt.savefig(f"{log_dir}/loss.jpg")
        plt.clf()

        # Duration
        plt.title("Duration")
        plt.xlabel("episode")
        plt.ylabel("duration")
        plt.plot(self.duration)
        plt.savefig(f"{log_dir}/duration.jpg")
        plt.clf()

        # Max Q value
        plt.title("Max Q value")
        plt.xlabel("step")
        plt.ylabel("Q value")
        plt.plot(self.max_q)
        plt.savefig(f"{log_dir}/max_q.jpg")
        plt.clf()

        # Game result
        plt.title("Result")
        plt.plot(self.win, "r.")
        plt.savefig(f"{log_dir}/result.jpg")
        plt.clf()
