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
    """
    A Gymnasium environment for Bullet Chess (1+0 time control).
    
    This environment simulates a game of chess where decision time is critical.
    The agent plays as White, and a simulated random opponent plays as Black.
    
    Attributes:
        board (chess.Board): The current state of the chess board.
        white_time (float): Remaining time for White (agent) in seconds.
        black_time (float): Remaining time for Black (opponent) in seconds.
        initial_time (float): The starting time for the game (default 60.0s).
    """
    
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode=None):
        """
        Initialize the Bullet Chess environment.

        Args:
            render_mode (str, optional): The render mode to use. Defaults to None.
                                         Supported modes: "ansi" (returns text representation).
        """
        self.render_mode = render_mode
        self.board = chess.Board()

        # Action: 4096 (from_sq * 64 + to_sq)
        # We use a flat discrete space representing all possible start->end square combinations.
        self.action_space = spaces.Discrete(64 * 64)

        # Observation: Dict
        # board: 8x8x12 (Pieces layers)
        # state: 8 (Globals: Turn, Castling Rights, En Passant, Normalized Time)
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
        """
        Reset the environment to the initial chess state.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for reset.

        Returns:
            obs (dict): The initial observation dictionary.
            info (dict): Auxiliary information (empty).
        """
        super().reset(seed=seed)
        self.board.reset()
        self.white_time = self.initial_time
        self.black_time = self.initial_time
        return self._get_obs(), {}

    def step(self, action):
        """
        Execute a step in the environment.

        The action can be a simple integer (standard Gymnasium) or a tuple
        (action_int, elapsed_time_float) if using the RealTimeClock wrapper.

        Args:
            action (int | tuple): The move to make.
                - If int: Represents the move index (0-4095). Time cost is assumed 0.0.
                - If tuple: (move_index, time_elapsed). usage with RealTimeClock.

        Returns:
            obs (dict): New observation.
            reward (float): Reward for the step (+1 win, -1 loss, 0 draw/continue, -10 illegal).
            terminated (bool): True if the game is over (checkmate/timeout).
            truncated (bool): Not currently used (always False).
            info (dict): Additional info (e.g., reason for game over).
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
            # Agent loses immediately for playing an illegal move to enforce valid play.
            return self._get_obs(), -10.0, True, False, {"error": "illegal_move"}

        # 4. Apply Agent Move
        self.board.push(move)

        if self.board.is_game_over():
            return self._handle_game_over()

        # 5. Opponent Move (Black - Random)
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            # Should be handled by is_game_over, but safety check for stalemate/mate conditions
            return self._handle_game_over()

        # Use Gymnasium's random generator for reproducibility
        opp_move = self.np_random.choice(legal_moves)
        self.board.push(opp_move)

        # Decrement Opponent Clock (Simulated)
        # We simulate a random thinking time for the opponent between 0.1s and 0.5s
        opp_cost = self.np_random.uniform(0.1, 0.5)
        self.black_time -= opp_cost

        if self.black_time <= 0:
            return self._get_obs(), 1.0, True, False, {"reason": "opponent_timeout"}

        if self.board.is_game_over():
            return self._handle_game_over()

        # 6. Continue
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        """
        Internal helper to construct the observation dictionary.
        """
        return {
            "board": get_board_tensor(self.board),
            "state": get_state_vector(
                self.board, self.white_time, self.black_time, self.initial_time
            ),
        }

    def _handle_game_over(self):
        """
        Internal helper to determine the result and reward when the game ends.
        """
        result = self.board.result()  # "1-0", "0-1", "1/2-1/2"
        reward = 0.0
        if result == "1-0":
            reward = 1.0
        elif result == "0-1":
            reward = -1.0
        # Draw is 0.0

        return self._get_obs(), reward, True, False, {"result": result}

    def render(self):
        """
        Render the environment.
        
        Returns:
            str: An ANSI string representation of the board if mode is "ansi".
        """
        if self.render_mode == "ansi":
            return str(self.board)
