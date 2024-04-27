import unittest

import torch

from model import MineSweeperCNN


class ModelTest(unittest.TestCase):

    def test_init(self):
        print("init_test")

        model = MineSweeperCNN(board_size=90, n_channels=32, depth=3)

        self.assertEqual(model._input.in_channels, 1)
        self.assertEqual(model._input.out_channels, 32)
        self.assertEqual(model._input.kernel_size, (3, 3))
        self.assertEqual(model._input.padding, "same")

        self.assertEqual(len(model._cnns), 3)
        for cnn in model._cnns:
            self.assertEqual(cnn.in_channels, 32)
            self.assertEqual(cnn.out_channels, 32)
            self.assertEqual(cnn.kernel_size, (3, 3))
            self.assertEqual(cnn.padding, "same")

        self.assertEqual(model._ff.in_features, 32 * 90)
        self.assertEqual(model._ff.out_features, 90)

    def test_forward(self):
        print("forward_test")

        model = MineSweeperCNN(board_size=90, n_channels=32, depth=3)

        state = torch.randn(5, 9, 10)
        output = model(state)

        self.assertEqual(output.size(0), 5)
        self.assertEqual(output.size(1), 90)


if __name__ == "__main__":
    unittest.main()
