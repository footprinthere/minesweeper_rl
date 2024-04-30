import unittest
from unittest.mock import MagicMock

from trainer import MineSweeperTrainer, TrainParameter
from game import MineSweeperEnv
from model.parameter import ModelParameter


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.env = MineSweeperEnv(9, 10, 10)
        self.env.reset(seed=1)

        self.model_param = ModelParameter(
            board_size=9 * 10, n_channels=2, cnn_depth=2, ff_dim=32
        )
        self.train_param = TrainParameter(batch_size=8)

    def test_select_action(self):
        trainer = MineSweeperTrainer(
            env=self.env, model_param=self.model_param, train_param=self.train_param
        )

        for _ in range(100):
            action = trainer._select_action()
            self.assertTrue(0 <= action < 9 * 10)

    def test_select_action_explore(self):
        self.train_param.eps_range = (1.0, 1.0)
        trainer = MineSweeperTrainer(
            env=self.env, model_param=self.model_param, train_param=self.train_param
        )
        self.env.sample_action = MagicMock()

        _ = trainer._select_action()
        self.env.sample_action.assert_called_once()

    def test_select_action_exploit(self):
        self.train_param.eps_range = (0.0, 0.0)
        trainer = MineSweeperTrainer(
            env=self.env, model_param=self.model_param, train_param=self.train_param
        )
        self.env.sample_action = MagicMock()

        _ = trainer._select_action()
        self.env.sample_action.assert_not_called()

    def test_train(self):
        trainer = MineSweeperTrainer(
            env=self.env, model_param=self.model_param, train_param=self.train_param
        )
        trainer.train(n_episodes=2)

        self.assertEqual(len(trainer.logs.loss), 2)
        self.assertEqual(len(trainer.logs.duration), 2)
        self.assertEqual(len(trainer.logs.win), 2)
        self.assertEqual(len(trainer.logs.max_q), 2)


if __name__ == "__main__":
    unittest.main()
