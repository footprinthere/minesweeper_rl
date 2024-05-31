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

from game import MineSweeperEnv, OpenResult
from model import MineSweeperCNN, ModelParameter
from trainer.memory import ReplayMemory, Transition
from trainer.q_tracker import MaxQValTracker
from trainer.parameter import TrainParameter
from tools.visualize import visualize_2d_tensor


class MineSweeperTrainer:

    def __init__(
        self,
        env: MineSweeperEnv,
        model_param: ModelParameter,
        train_param: TrainParameter,
        device: torch.device,
        project_dir: Optional[str] = None,
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
            device=device,
            use_mask=self.train_param.use_action_mask,
        )
        self.q_tracker.collect_samples(size=self.train_param.q_sample_size)

        self.device = device
        self.project_dir = project_dir

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def train(self, n_episodes: int) -> None:
        self.logs.clear()

        for i in tqdm(range(n_episodes)):
            # Reset environment
            self.env.reset()

            for t in count():
                win = self._step()
                if self.env.is_terminated():
                    break  # episode terminated

            # Save metrics
            max_q, _ = self.q_tracker.get_max_q()
            self.logs.episode(i, duration=t + 1, max_q=max_q, win=win)

    def _step(self) -> bool:
        prev_state = self.env.get_state()
        assert prev_state is not None
        action = self._select_action(use_mask=self.train_param.use_action_mask)

        env_step_result = self.env.step(action)
        next_state = self.env.get_state()

        # Store transition in memory
        self.memory.push(
            state=prev_state,
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
        self.logs.step_reward(env_step_result.reward)
        if loss is not None:
            self.logs.step_loss(loss)

        win = env_step_result.open_result == OpenResult.WIN
        return win

    def _optimize(self) -> Optional[float]:
        if self.memory.size < self.train_param.batch_size:
            return None

        transitions = self.memory.sample(self.train_param.batch_size)
        batch, non_final_mask = Transition.concat(transitions)
        batch.to_device(self.device)

        # Compute Q(s_t, a)
        #   The policy network returns Q(s),
        #   and then we choose the values corresponding to the given actions
        q = self.policy_net(batch.state)  # [batch_size, n_actions]
        state_action_values = torch.gather(q, dim=1, index=batch.action.unsqueeze(1))
        # [batch_size, 1]

        # Compute V(s_{t+1}) = R + γ * max_a Q(s_{t+1}, a)
        next_state_values = torch.zeros(self.train_param.batch_size, device=self.device)
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

    def _select_action(self, use_mask: bool = False) -> int:
        p = random.random()
        eps_start, eps_end = self.train_param.eps_range
        eps = eps_end + (eps_start - eps_end) * math.exp(
            -1.0 * self.steps_done / self.train_param.eps_decay
        )
        self.logs.step_eps(eps)
        if p < eps:
            # Explore
            return self.env.sample_action(exclude_opened=use_mask)

        # Exploit
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

    def save_models(self) -> None:
        if self.project_dir is not None:
            path_format = os.path.join(self.project_dir, "{}.pt")
        else:
            path_format = "{}.pt"
        self.policy_net.save_state(path_format.format("policy"))
        self.target_net.save_state(path_format.format("target"))

    def load_models(self, directory: Optional[str] = None) -> None:
        """If directory is None, it loads the models from the project_dir"""
        if directory is None:
            if self.project_dir is None:
                raise ValueError("project_dir is not set")
            directory = self.project_dir

        path_format = os.path.join(directory, "{}.pt")
        self.policy_net.load_state(path_format.format("policy"))
        self.target_net.load_state(path_format.format("target"))

    def plot_logs(self, average_episodes: int) -> None:
        if self.project_dir is None:
            raise ValueError("project_dir is not set")
        averaged = self.logs.average_every_n_episodes(n=average_episodes)
        averaged.plot(self.project_dir)

    def visualize_q_values(
        self,
        use_mask: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        output = self._compute_q(use_mask=use_mask).cpu()
        output = output.reshape(self.env.board_height, self.env.board_width)
        visualize_2d_tensor(output, lower_bound=-1e8, title=title, save_path=save_path)


@dataclass
class TrainLog:
    duration: list[int] = field(default_factory=list)
    max_q: list[float] = field(default_factory=list)
    win: list[bool] = field(default_factory=list)

    loss: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    _step_loss: list[float] = field(default_factory=list)
    _step_reward: list[float] = field(default_factory=list)

    eps: list[float] = field(default_factory=list)

    def clear(self):
        self.duration.clear()
        self.max_q.clear()
        self.win.clear()
        self.loss.clear()
        self.reward.clear()
        self._step_loss.clear()
        self._step_reward.clear()
        self.eps.clear()

    def episode(self, episode_idx: int, duration: int, max_q: float, win: bool) -> None:
        self.duration.append(duration)
        self.max_q.append(max_q)
        self.win.append(win)

        self.loss.append(sum(self._step_loss) / duration)
        self.reward.append(avg_reward := sum(self._step_reward) / duration)
        self._step_loss.clear()
        self._step_reward.clear()

        if (episode_idx + 1) % 100 == 0:
            tqdm.write(
                f"Episode {episode_idx+1} : duration {duration}, reward {avg_reward}, max_q {max_q}, win {win}"
            )

    def step_loss(self, loss: float) -> None:
        self._step_loss.append(loss)

    def step_reward(self, reward: float) -> None:
        self._step_reward.append(reward)

    def step_eps(self, eps: float) -> None:
        self.eps.append(eps)

    def average_every_n_episodes(self, n: int) -> "TrainLog":
        def average(arr: list[float]) -> list[float]:
            return [sum(arr[i : i + n]) / n for i in range(0, len(arr), n)]

        return TrainLog(
            duration=[e for e in self.duration],
            max_q=average(self.max_q),
            win=[e for e in self.win],
            loss=average(self.loss),
            reward=average(self.reward),
            eps=average(self.eps),
        )

    def plot(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        name_map = {
            "duration": self.duration,
            "max_q": self.max_q,
            "loss": self.loss,
            "reward": self.reward,
            "eps": self.eps,
        }

        for name, log in name_map.items():
            plt.title(name.upper())
            plt.plot(log)
            plt.savefig(f"{directory}/{name}.jpg")
            plt.clf()

        # Game result
        plt.title(f"WIN ({self.win.count(True)}/{len(self.win)})")
        plt.plot(self.win, "r.")
        plt.savefig(f"{directory}/win.jpg")
        plt.clf()
