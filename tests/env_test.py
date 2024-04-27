import unittest

from game import MineSweeperEnv
from game.board import MineSweeperBoard
from game.open_result import OpenResult


class EnvTest(unittest.TestCase):

    H = 9
    W = 10
    N_MINES = 10

    @classmethod
    def setUpClass(cls):
        print("setUpClass")

        cls.env = MineSweeperEnv(cls.H, cls.W, cls.N_MINES)

    def setUp(self):
        self.env.reset(seed=1)

    def test_reset(self):
        print("test_reset")

        board = self.env.reset(seed=1)
        self.assertEqual(len(board), self.H)
        self.assertEqual(len(board[0]), self.W)

        for row in board:
            for cell in row:
                self.assertEqual(cell, MineSweeperBoard.CLOSED)
        for row in self.env.get_open_state():
            for cell in row:
                self.assertFalse(cell)

    def test_step_first(self):
        print("test_step_first")

        result = self.env.step(4)

        self.assertNotEqual(result.visible_board[0][4], MineSweeperBoard.CLOSED)
        self.assertEqual(result.reward, -1)
        self.assertFalse(result.terminated)
        self.assertEqual(result.open_result, OpenResult.ISOLATED)
        self.assertTrue(self.env.get_open_state()[0][4])

    def test_step_adjacent(self):
        print("test_step_adjacent")

        result = self.env.step(4)
        result = self.env.step(5)

        self.assertNotEqual(result.visible_board[0][5], MineSweeperBoard.CLOSED)
        self.assertEqual(result.reward, 1)
        self.assertFalse(result.terminated)
        self.assertEqual(result.open_result, OpenResult.ADJACENT)
        self.assertTrue(self.env.get_open_state()[0][5])

    def test_step_mine(self):
        print("test_step_mine")

        result = self.env.step(8)

        self.assertEqual(result.reward, -10)
        self.assertTrue(result.terminated)
        self.assertEqual(result.open_result, OpenResult.FAIL)

    def test_sample_action(self):
        print("test_sample_action")

        action = self.env.sample_action()
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.H * self.W)


if __name__ == "__main__":
    unittest.main()
