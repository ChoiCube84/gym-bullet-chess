import unittest
import gymnasium as gym
import time
import gym_bullet_chess
from gym_bullet_chess.wrappers.real_time import RealTimeClock


class TestRealTime(unittest.TestCase):
    def setUp(self):
        # Create the base environment
        self.base_env = gym.make("BulletChess-v0")
        # Create the wrapped environment
        self.rt_env = RealTimeClock(gym.make("BulletChess-v0"))

    def test_case_a_real_time_wrapper(self):
        """Test Case A: Verify RealTimeClock wrapper deducts elapsed time."""
        self.rt_env.reset()

        # Valid move: White e2e4 (12 -> 28)
        action = 12 * 64 + 28

        # Sleep for 1.0 seconds
        time.sleep(1.0)

        obs, reward, terminated, truncated, info = self.rt_env.step(action)

        # Initial time is 60.0.
        # Check if white_time (index 6 in state vector) has decreased by approx 1.0s.
        # State[6] is normalized time (time / 60.0).
        # So remaining time should be approx 59.0.
        # obs["state"][6] should be approx 59.0 / 60.0.

        remaining_time_norm = obs["state"][6]
        remaining_time = remaining_time_norm * 60.0

        # We allow some tolerance because time.sleep is not perfect and there's execution overhead
        self.assertLess(remaining_time, 59.2)
        self.assertGreater(
            remaining_time, 58.0
        )  # Should definitely be less than 60, and around 59

    def test_case_b_core_env_explicit_time(self):
        """Test Case B: Verify Core Env handles explicit time deduction."""
        self.base_env.reset()

        # Valid move: White e2e4
        action_idx = 12 * 64 + 28
        explicit_time_deduction = 5.0

        # Pass tuple (action, time)
        obs, reward, terminated, truncated, info = self.base_env.step(
            (action_idx, explicit_time_deduction)
        )

        remaining_time_norm = obs["state"][6]
        remaining_time = remaining_time_norm * 60.0

        # Should be exactly 60.0 - 5.0 = 55.0
        self.assertAlmostEqual(remaining_time, 55.0, places=4)

    def test_case_c_timeout(self):
        """Test Case C: Verify Timeout behavior."""
        self.base_env.reset()

        # Valid move: White e2e4
        action_idx = 12 * 64 + 28

        # Deduct 61 seconds (more than 60s initial time)
        obs, reward, terminated, truncated, info = self.base_env.step(
            (action_idx, 61.0)
        )

        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info.get("reason"), "timeout")

    def test_compatibility_reset(self):
        """Verify env.reset() works with wrapper."""
        obs, info = self.rt_env.reset()
        self.assertIn("board", obs)
        self.assertIn("state", obs)
        self.assertIsInstance(info, dict)

        # Check initial time is full
        self.assertAlmostEqual(obs["state"][6], 1.0)

    def test_compatibility_step_return(self):
        """Verify env.step() returns correct 5-tuple."""
        self.rt_env.reset()
        action = 12 * 64 + 28
        ret = self.rt_env.step(action)
        self.assertEqual(len(ret), 5)
        obs, reward, terminated, truncated, info = ret
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_edge_case_huge_time(self):
        """Edge Case: Huge elapsed time."""
        self.base_env.reset()
        action = 12 * 64 + 28
        # Deduct 1000 seconds
        obs, reward, terminated, truncated, info = self.base_env.step((action, 1000.0))

        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info.get("reason"), "timeout")

        # Time should be 0 (normalized)
        self.assertEqual(obs["state"][6], 0.0)

    def test_edge_case_zero_time(self):
        """Edge Case: 0.0 elapsed time."""
        self.base_env.reset()
        action = 12 * 64 + 28
        obs, reward, terminated, truncated, info = self.base_env.step((action, 0.0))

        self.assertFalse(terminated)
        # Time should be 60.0
        self.assertAlmostEqual(obs["state"][6] * 60.0, 60.0)

    def test_edge_case_invalid_tuple(self):
        """Edge Case: Invalid tuple formats."""
        self.base_env.reset()

        # Case 1: Tuple too short -> IndexError expected based on code inspection
        # But we want to see what happens. Ideally it should be handled or raise a clear error.
        # The current code: move_idx = action[0], elapsed = float(action[1])
        with self.assertRaises(IndexError):
            self.base_env.step((123,))

        # Case 2: Tuple too long -> Should work, taking first two elements
        self.base_env.reset()
        action = 12 * 64 + 28
        obs, reward, terminated, truncated, info = self.base_env.step(
            (action, 5.0, "extra_junk")
        )
        self.assertFalse(terminated)
        self.assertAlmostEqual(obs["state"][6] * 60.0, 55.0)


if __name__ == "__main__":
    unittest.main()
