import gymnasium as gym
import numpy as np
import chess
import random
from gymnasium import spaces
from gym_bullet_chess.utils.encoding import (
    get_board_tensor,
    get_state_vector,
    int_to_move,
)


class BulletChessEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.board = chess.Board()

        # Action: 4096 (from_sq * 64 + to_sq)
        self.action_space = spaces.Discrete(64 * 64)

        # Observation: Dict
        # board: 8x8x12 (Pieces)
        # state: 8 (Globals: Turn, Castling, EP, Time)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32),
                "state": spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32),
            }
        )

        # Time Control (Bullet 1+0: 60 seconds)
        self.initial_time = 60.0
        self.white_time = self.initial_time
        self.black_time = self.initial_time

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        self.white_time = self.initial_time
        self.black_time = self.initial_time
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes a move.
        Args:
            action: Either an int (standard) or a tuple (int, time_elapsed).
                    If int, time_elapsed defaults to 0.0.
        """
        # 0. Parse Action and Time
        elapsed_time = 0.0
        move_idx = action

        if isinstance(action, (tuple, list)):
            move_idx = action[0]
            elapsed_time = float(action[1])

        # 1. Decrement Agent Clock (Based on input)
        self.white_time -= elapsed_time

        # 2. Check Agent Termination (Timeout)
        if self.white_time <= 0:
            return self._get_obs(), -1.0, True, False, {"reason": "timeout"}

        # 3. Decode and Validate Agent Move
        move = int_to_move(move_idx, self.board)

        if move not in self.board.legal_moves:
            # Illegal move penalty
            return self._get_obs(), -10.0, True, False, {"error": "illegal_move"}

        # 4. Apply Agent Move
        self.board.push(move)

        if self.board.is_game_over():
            return self._handle_game_over()

        # 5. Opponent Move (Black - Random)
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return self._handle_game_over()

        # Use Gymnasium's random generator for reproducibility
        opp_move = self.np_random.choice(legal_moves)
        self.board.push(opp_move)

        # Decrement Opponent Clock (Simulated)
        opp_cost = self.np_random.uniform(0.1, 0.5)
        self.black_time -= opp_cost

        if self.black_time <= 0:
            return self._get_obs(), 1.0, True, False, {"reason": "opponent_timeout"}

        if self.board.is_game_over():
            return self._handle_game_over()

        # 6. Continue
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        return {
            "board": get_board_tensor(self.board),
            "state": get_state_vector(
                self.board, self.white_time, self.black_time, self.initial_time
            ),
        }

    def _handle_game_over(self):
        result = self.board.result()  # "1-0", "0-1", "1/2-1/2"
        reward = 0.0
        if result == "1-0":
            reward = 1.0
        elif result == "0-1":
            reward = -1.0
        # Draw is 0.0

        return self._get_obs(), reward, True, False, {"result": result}

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
