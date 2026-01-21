import gymnasium as gym
import numpy as np
import chess
import pygame
import os
from gymnasium import spaces
from gym_bullet_chess.utils.encoding import (
    get_board_tensor,
    get_state_vector,
    int_to_move,
)


class BulletChessEnv(gym.Env):
    """
    A Gymnasium environment for Bullet Chess with flexible time controls.

    The agent plays as White, and a simulated random opponent plays as Black.

    Attributes:
        board (chess.Board): The current state of the chess board.
        white_time (float): Remaining time for White (agent) in seconds.
        black_time (float): Remaining time for Black (opponent) in seconds.
        time_limit (float): The starting time for the game (e.g., 60.0s).
        increment (float): Time added after each move (e.g., 0.0s or 2.0s).
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        self_play=False,
        time_limit=60.0,
        increment=0.0,
        capture_visual=False,
    ):
        """
        Initialize the Bullet Chess environment.

        Args:
            render_mode (str, optional): "ansi" (text) or "rgb_array" (image).
            self_play (bool, optional): If True, agent controls both colors.
            time_limit (float, optional): Initial time in seconds (default 60.0).
            increment (float, optional): Time increment per move in seconds (default 0.0).
            capture_visual (bool, optional): If True, observation includes 'board_img'.
        """
        self.render_mode = render_mode
        self.self_play = self_play
        self.time_limit = float(time_limit)
        self.increment = float(increment)
        self.capture_visual = capture_visual

        self.board = chess.Board()
        self.window = None
        self.clock = None
        self.canvas = None  # Reusable surface
        self.window_size = 512  # Size of the Pygame window

        # Asset Loading
        self.piece_images = {}
        self.assets_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "assets"
        )
        self.assets_loaded = False

        # Action: 4096 (from_sq * 64 + to_sq)
        self.action_space = spaces.Discrete(64 * 64)

        # Observation Space construction
        obs_dict = {
            "board": spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32),
            "state": spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32),
        }

        if self.capture_visual:
            obs_dict["board_img"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.window_size, self.window_size, 3),
                dtype=np.uint8,
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Clocks
        self.white_time = self.time_limit
        self.black_time = self.time_limit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()

        # Reset Clocks
        self.white_time = self.time_limit
        self.black_time = self.time_limit

        # Handle options
        if options:
            if "self_play" in options:
                self.self_play = options["self_play"]

        return self._get_obs(), {}

    def step(self, action):
        # 0. Parse Action and Time
        elapsed_time = 0.0
        move_idx = action

        # Robust unpacking (handles nested tuples if wrappers stack)
        while isinstance(move_idx, (tuple, list)):
            if len(move_idx) >= 2:
                # Assuming (action, time) pattern
                elapsed_time += float(move_idx[1])
                move_idx = move_idx[0]
            elif len(move_idx) == 1:
                move_idx = move_idx[0]
            else:
                # Empty tuple?
                break

        if not np.isfinite(elapsed_time) or elapsed_time < 0:
            elapsed_time = 0.0

        # 1. Decrement Clock (Current Turn)
        is_white_turn = self.board.turn == chess.WHITE

        if is_white_turn:
            self.white_time -= elapsed_time
            if self.white_time <= 0:
                # Timeout Handling: Soft reset for API model testing
                # Instead of ending the game, we reset the clock and continue.
                # This allows us to benchmark move quality even if the API is slow.
                self.white_time = self.time_limit  # Reset clock
                return self._get_obs(), 0.0, False, False, {"warning": "timeout_reset_white"}
        else:
            self.black_time -= elapsed_time
            if self.black_time <= 0:
                # Timeout Handling: Soft reset for API model testing
                self.black_time = self.time_limit  # Reset clock
                return self._get_obs(), 0.0, False, False, {"warning": "timeout_reset_black"}

        # 3. Decode and Validate Move
        # Ensure integer
        try:
            move_idx = int(move_idx)
        except (ValueError, TypeError):
            return (
                self._get_obs(),
                -10.0,
                True,
                False,
                {"error": "invalid_action_format"},
            )

        if not (0 <= move_idx < self.action_space.n):
            return (
                self._get_obs(),
                -10.0,
                True,
                False,
                {"error": "action_out_of_bounds"},
            )

        move = int_to_move(move_idx, self.board)

        if move not in self.board.legal_moves:
            # Illegal move: Immediate loss, NO increment.
            return self._get_obs(), -10.0, True, False, {"error": "illegal_move"}

        # 4. Apply Move
        self.board.push(move)

        # 5. Apply Increment (Only after legal move)
        if is_white_turn:
            self.white_time += self.increment
        else:
            self.black_time += self.increment

        if self.board.is_game_over():
            return self._handle_game_over()

        if self.self_play:
            # Self-play: return 0.0 intermediate reward
            return self._get_obs(), 0.0, False, False, {}

        # 6. Opponent Move (Black - Random)
        # Only if we are in Single Player mode and it is now Black's turn
        if self.board.turn == chess.BLACK:
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                return self._handle_game_over()

            opp_move = self.np_random.choice(legal_moves)

            # Opponent thinks...
            min_think = 0.1
            max_think = max(0.5, self.time_limit * 0.02)
            opp_cost = self.np_random.uniform(min_think, max_think)

            self.black_time -= opp_cost
            if self.black_time <= 0:
                return self._get_obs(), 1.0, True, False, {"reason": "opponent_timeout"}

            self.board.push(opp_move)
            self.black_time += self.increment

            if self.board.is_game_over():
                return self._handle_game_over()

        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        obs = {
            "board": get_board_tensor(self.board),
            "state": get_state_vector(
                self.board, self.white_time, self.black_time, self.time_limit
            ),
        }

        if self.capture_visual:
            obs["board_img"] = self._render_frame()

        return obs

    def _handle_game_over(self):
        result = self.board.result()
        # Results: "1-0", "0-1", "1/2-1/2"

        # Base reward from White's perspective
        reward = 0.0
        if result == "1-0":
            reward = 1.0
        elif result == "0-1":
            reward = -1.0

        if self.self_play:
            # If self-play, the reward should be for the player who Just Moved.
            # self.board.turn is now the player who needs to move NEXT (but can't, or game over).
            # So if it's White's turn now, Black just moved.

            if self.board.turn == chess.WHITE:
                # Black just moved.
                # If Black won ("0-1", reward=-1.0), we want +1.0 for Black.
                reward = -reward
            # If it's Black's turn now, White just moved.
            # If White won ("1-0", reward=1.0), we want +1.0 for White.
            # So reward stays as is.

        return self._get_obs(), reward, True, False, {"result": result}

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _load_assets(self):
        """Loads piece images from assets directory."""
        if self.assets_loaded:
            return

        square_size = self.window_size // 8

        piece_map = {
            "P": "wP",
            "N": "wN",
            "B": "wB",
            "R": "wR",
            "Q": "wQ",
            "K": "wK",
            "p": "bP",
            "n": "bN",
            "b": "bB",
            "r": "bR",
            "q": "bQ",
            "k": "bK",
        }

        for symbol, fname in piece_map.items():
            path = os.path.join(self.assets_dir, f"{fname}.png")
            if os.path.exists(path):
                img = pygame.image.load(path)
                img = pygame.transform.smoothscale(img, (square_size, square_size))
                self.piece_images[symbol] = img
            else:
                print(f"Warning: Asset not found {path}")

        self.assets_loaded = True

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Reuse surface if possible
        if self.canvas is None:
            self.canvas = pygame.Surface((self.window_size, self.window_size))

        if not self.assets_loaded:
            if not pygame.get_init():
                pygame.init()
            self._load_assets()

        light_sq = (240, 217, 181)
        dark_sq = (181, 136, 99)
        square_size = self.window_size // 8

        # Fallback font
        if not self.piece_images:
            if not pygame.font.get_init():
                pygame.font.init()
            try:
                font = pygame.font.SysFont("dejavusans", int(square_size * 0.8))
            except:
                font = pygame.font.Font(None, int(square_size * 0.8))

        is_white_view = self.board.turn == chess.WHITE

        for r in range(8):
            for c in range(8):
                # Draw Square
                color = light_sq if (r + c) % 2 == 0 else dark_sq
                rect = pygame.Rect(
                    c * square_size, r * square_size, square_size, square_size
                )
                pygame.draw.rect(self.canvas, color, rect)

                if is_white_view:
                    rank = 7 - r
                    file = c
                else:
                    rank = r
                    file = 7 - c

                square_idx = chess.square(file, rank)
                piece = self.board.piece_at(square_idx)

                if piece:
                    symbol = piece.symbol()

                    if symbol in self.piece_images:
                        self.canvas.blit(self.piece_images[symbol], rect)
                    else:
                        # Fallback text rendering
                        pass  # Kept simple for now

        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
        )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
