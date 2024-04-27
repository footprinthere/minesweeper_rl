import unittest
from unittest.mock import MagicMock

import torch

from train import MineSweeperTrainer
from game import MineSweeperEnv


class TrainerTest(unittest.TestCase):

    trainer_params = {
        "board_size": 9 * 10,
        "n_channels": 2,
        "model_depth": 2,
    }

    def setUp(self):
        self.env = MineSweeperEnv(9, 10, 10)
        self.init_board = self.env.reset(seed=1)
        self.init_state = torch.tensor(self.init_board, dtype=torch.float32).unsqueeze(
            0
        )

    def test_select_action(self):
        trainer = MineSweeperTrainer(env=self.env, **self.trainer_params)

        for _ in range(100):
            action = trainer._select_action(self.init_state)
            self.assertTrue(0 <= action < 9 * 10)

    def test_select_action_explore(self):
        self.env.sample_action = MagicMock()
        trainer = MineSweeperTrainer(
            env=self.env, eps_range=(1.0, 1.0), **self.trainer_params
        )

        _ = trainer._select_action(self.init_state)
        self.env.sample_action.assert_called_once()

    def test_select_action_exploit(self):
        self.env.sample_action = MagicMock()
        trainer = MineSweeperTrainer(
            env=self.env, eps_range=(0.0, 0.0), **self.trainer_params
        )

        _ = trainer._select_action(self.init_state)
        self.env.sample_action.assert_not_called()

    def test_train(self):
        trainer = MineSweeperTrainer(env=self.env, **self.trainer_params)
        trainer.train(n_episodes=2)

        self.assertEqual(len(trainer.logs.loss), 2)
        self.assertEqual(len(trainer.logs.duration), 2)
        self.assertEqual(len(trainer.logs.win), 2)
        self.assertTrue(len(trainer.logs.max_q) > 2)


if __name__ == "__main__":
    unittest.main()
