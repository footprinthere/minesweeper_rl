from typing import Optional
from dataclasses import dataclass, field
from itertools import count
import math
import random
import os

import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MineSweeperCNN, ModelParameter
from game.env import MineSweeperEnv
from game.open_result import OpenResult
from .memory import ReplayMemory, Transition
from .train_step_result import TrainStepResult
from .q_tracker import MaxQValTracker
from .parameter import TrainParameter


class MineSweeperTrainer:

    def __init__(
        self,
        env: MineSweeperEnv,
        model_param: ModelParameter,
        train_param: TrainParameter,
    ):
        self.env = env

        self.policy_net = MineSweeperCNN.with_parameter(model_param)
        self.target_net = MineSweeperCNN.with_parameter(model_param)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.train_param = train_param
        self.memory = ReplayMemory(capacity=self.train_param.memory_size)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.train_param.lr
        )

        self.steps_done = 0
        self.logs = TrainLog()
        self.q_tracker = MaxQValTracker(
            env=env,
            policy_net=self.policy_net,
            use_mask=self.train_param.use_action_mask,
        )
        self.q_tracker.collect_samples(size=self.train_param.q_sample_size)

    def train(self, n_episodes: int, log_file: Optional[str] = None) -> None:
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

                if log_file is not None:
                    raise NotImplementedError
                else:
                    tqdm.write(self.env.render())

                if step_result.next_state is not None:
                    state = step_result.next_state
                else:
                    break  # episode terminated

            # Save metrics
            episode_loss /= t + 1
            self.logs.loss.append(episode_loss)
            self.logs.duration.append(t + 1)
            self.logs.win.append(step_result.open_result == OpenResult.WIN)

            max_q, max_q_softmax = self.q_tracker.get_max_q()
            self.logs.max_q.append(max_q)
            self.logs.max_q_softmax.append(max_q_softmax)

            tqdm.write(f"Episode {i} - loss {episode_loss :.4f}, duration {t + 1}")

    def save_models(self, path: str) -> None:
        raise NotImplementedError

    def plot_logs(self, log_dir: str) -> None:
        self.logs.plot(log_dir)

    def _step(self, state: Tensor) -> TrainStepResult:
        action = self._select_action(state, use_mask=self.train_param.use_action_mask)
        env_step_result = self.env.step(action)
        if env_step_result.terminated:
            next_state = None
        else:
            next_state = torch.tensor(
                env_step_result.visible_board, dtype=torch.float32
            ).unsqueeze(0)

        # Store transition in memory
        self.memory.push(
            state=state,
            action=torch.tensor([action]),
            reward=torch.tensor([env_step_result.reward]),
            next_state=next_state,
        )

        # Perform one step of optimization on the policy network
        loss = self._optimize()

        # Soft update the target network's weights
        #   Alternatively, we can update the target network's weights every C steps
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.train_param.tau * policy_param.data
                + (1.0 - self.train_param.tau) * target_param.data
            )

        self.steps_done += 1
        return TrainStepResult(
            loss=loss,
            next_state=next_state,
            open_result=env_step_result.open_result,
        )

    def _optimize(self) -> Optional[float]:
        if self.memory.size < self.train_param.batch_size:
            return None

        transitions = self.memory.sample(self.train_param.batch_size)
        batch, non_final_mask = Transition.concat(transitions)

        # Compute Q(s_t, a)
        #   The policy network returns Q(s),
        #   and then we choose the values corresponding to the given actions
        q = self.policy_net(batch.state)  # [batch_size, n_actions]
        state_action_values = torch.gather(q, dim=1, index=batch.action.unsqueeze(1))
        # [batch_size, 1]

        # Compute V(s_{t+1}) = R + γ * max_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.train_param.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(
                self.target_net(batch.next_state), dim=1
            ).values
        expected_state_action_values = (
            next_state_values * self.train_param.gamma + batch.reward
        )

        # Compute loss
        criterion = torch.nn.SmoothL1Loss()
        loss: Tensor = criterion(
            state_action_values.squeeze(), expected_state_action_values
        )

        # Optimize policy network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.train_param.grad_max_norm
        )
        self.optimizer.step()

        return loss.item()

    def _select_action(self, state: Tensor, use_mask: bool = False) -> int:
        p = random.random()
        eps_start, eps_end = self.train_param.eps_range
        eps = eps_end + (eps_start - eps_end) * math.exp(
            -1.0 * self.steps_done / self.train_param.eps_decay
        )
        if p < eps:
            # Explore
            return self.env.sample_action(exclude_opened=use_mask)

        # Exploit
        with torch.no_grad():
            output = self.policy_net(state)
        if use_mask:
            mask = torch.tensor(self.env.get_open_state(), dtype=torch.bool).flatten()
            output = torch.masked_fill(output, mask=mask, value=-1e9)

        argmax = torch.max(output, dim=1).indices
        return int(argmax.item())


@dataclass
class TrainLog:
    loss: list[float] = field(default_factory=list)
    duration: list[int] = field(default_factory=list)
    win: list[bool] = field(default_factory=list)
    max_q: list[float] = field(default_factory=list)
    max_q_softmax: list[float] = field(default_factory=list)

    def clear(self):
        self.loss.clear()
        self.duration.clear()
        self.win.clear()
        self.max_q.clear()
        self.max_q_softmax.clear()

    def plot(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)

        name_map = {
            "loss": self.loss,
            "duration": self.duration,
            "max_q": self.max_q,
            "max_q_softmax": self.max_q_softmax,
        }

        for name, log in name_map.items():
            plt.title(name.upper())
            plt.xlabel("episode")
            plt.ylabel(name)
            plt.plot(log)
            plt.savefig(f"{log_dir}/{name}.jpg")
            plt.clf()

        # Game result
        plt.title("RESULT")
        plt.plot(self.win, "r.")
        plt.savefig(f"{log_dir}/result.jpg")
        plt.clf()
