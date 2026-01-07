import gymnasium as gym
import time


class RealTimeClock(gym.Wrapper):
    """
    A wrapper that measures the wall-clock time between the last observation (step/reset)
    and the current action (step), and deducts that duration from the agent's clock.

    This enforces "Real-Time" constraints on the agent. If the agent takes 10 seconds
    to compute an action, 10 seconds will be subtracted from its game clock.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_time = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start the clock immediately after reset returns
        self.last_time = time.perf_counter()
        return obs, info

    def step(self, action):
        if self.last_time is None:
            # Should not happen if reset was called, but safety first
            self.last_time = time.perf_counter()

        current_time = time.perf_counter()
        elapsed = current_time - self.last_time

        # Pass the tuple (action, elapsed) to the inner env
        # The inner BulletChessEnv is modified to handle this tuple.
        obs, reward, terminated, truncated, info = self.env.step((action, elapsed))

        # Restart the clock for the NEXT move
        self.last_time = time.perf_counter()

        return obs, reward, terminated, truncated, info
