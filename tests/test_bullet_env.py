import unittest
import gymnasium as gym
import numpy as np
import gym_bullet_chess
import chess


class TestBulletChessEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("BulletChess-v0")

    def test_env_make(self):
        # Just check we can access the unwrapped env
        self.assertTrue(hasattr(self.env.unwrapped, "board"))

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs["board"].shape, (8, 8, 12))
        self.assertEqual(obs["state"].shape, (8,))
        # Initial Time: 60/60 = 1.0
        self.assertAlmostEqual(obs["state"][6], 1.0)
        self.assertAlmostEqual(obs["state"][7], 1.0)

    def test_step_valid(self):
        self.env.reset()
        # White e2e4: e2(12) -> e4(28). 12*64+28 = 796
        action = 12 * 64 + 28
        obs, reward, term, trunc, info = self.env.step(action)
        self.assertFalse(term)
        self.assertEqual(reward, 0.0)
        # Check white time decreased
        # In the new architecture, passing an int implies 0.0 elapsed time for the agent.
        self.assertAlmostEqual(obs["state"][6], 1.0)

    def test_step_illegal(self):
        self.env.reset()
        # Illegal: a1a8 (0->56) - Rook blocked by Pawn
        action = 0 * 64 + 56
        obs, reward, term, trunc, info = self.env.step(action)
        self.assertTrue(term)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info["error"], "illegal_move")

    def test_checkmate_win(self):
        self.env.reset()
        board = self.env.unwrapped.board
        # FEN: "7k/Q7/5K2/8/8/8/8/8 w - - 0 1" (White K f6, Q a7. Black K h8).
        # Move Qg7# is mate.
        board.set_fen("7k/Q7/5K2/8/8/8/8/8 w - - 0 1")

        # Action: a7 (48) -> g7 (54)
        action = 48 * 64 + 54
        obs, reward, term, trunc, info = self.env.step(action)
        self.assertTrue(term)
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["result"], "1-0")


if __name__ == "__main__":
    unittest.main()
