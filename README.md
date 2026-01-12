# gym-bullet-chess

A Gymnasium-compatible bullet chess environment for reinforcement learning research under real-time decision constraints.

This environment models **bullet-style chess**, where agents must trade off move quality against limited decision time. Unlike standard chess environments, `gym-bullet-chess` explicitly includes time management as part of the state space and termination criteria.

This project is open-source, research-oriented, and not affiliated with any online chess platform.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ChoiCube84/gym-bullet-chess.git
cd gym-bullet-chess
pip install -e .
```

---

## Usage

### Basic Usage (1+0 Bullet)

```python
import gymnasium as gym
import gym_bullet_chess  # registers the environment

env = gym.make("BulletChess-v0")
obs, info = env.reset()

done = False
while not done:
    # Random action
    action = env.action_space.sample()
    
    # Step returns standard Gymnasium tuple
    obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated

env.close()
```

### Custom Time Controls (e.g., 3+2 Blitz)

You can configure the initial time and increment using `time_limit` (seconds) and `increment` (seconds).

```python
# 3 minutes initial time + 2 seconds increment per move
env = gym.make("BulletChess-v0", time_limit=180.0, increment=2.0)
```

### Visual Observations (VLM Support)

For Vision-Language Models (VLMs), you can enable visual observations. This returns a 512x512 RGB image of the board.

```python
env = gym.make("BulletChess-v0", capture_visual=True)
obs, info = env.reset()

# Access the image (Height, Width, 3)
board_image = obs["board_img"] 
```

**Note:** In self-play mode, the board automatically flips perspective (180°) when it is Black's turn, ensuring the agent always sees the board from its own perspective.

### Self-Play Mode

You can enable self-play mode to control both White and Black pieces. This disables the automatic random opponent.

```python
# Enable self-play at initialization
env = gym.make("BulletChess-v0", self_play=True)

obs, info = env.reset()

# Play a move for White
obs, reward, terminated, truncated, info = env.step(white_action)

if not terminated:
    # Play a move for Black (Observation will be FLIPPED for Black's perspective)
    obs, reward, terminated, truncated, info = env.step(black_action)
```

### Real-Time Constraints

To properly simulate bullet chess, you should use the `RealTimeClock` wrapper. This wrapper measures the time your agent takes to compute an action and deducts it from the in-game clock.

```python
from gym_bullet_chess.wrappers import RealTimeClock

# 1. Create environment
env = gym.make("BulletChess-v0")

# 2. Wrap it to enforce real-time constraints
env = RealTimeClock(env)

obs, info = env.reset()

# If this loop takes 5 seconds of wall-clock time, 
# 5 seconds are removed from the agent's game clock.
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

---

## Environment Details

### Observation Space

The observation is a `Dict` space.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `board` | `(8, 8, 12)` | `float32` | 8x8 spatial representation (One-Hot per piece type). |
| `state` | `(8,)` | `float32` | Global state vector containing flags and time info. |
| `board_img` | `(512, 512, 3)` | `uint8` | **(Optional)** RGB image of the board if `capture_visual=True`. |

**State Vector Layout:**
- Index 0: **Turn** (1.0 = White, 0.0 = Black)
- Index 1-4: **Castling Rights** (White King/Queen, Black King/Queen)
- Index 5: **En Passant** (1.0 if available)
- Index 6: **White Time** (Normalized: `current / time_limit`. Can be > 1.0 with increment)
- Index 7: **Black Time** (Normalized: `current / time_limit`. Can be > 1.0 with increment)

### Action Space

The action space is `Discrete(4096)`. 
Each integer action represents a move `from_square * 64 + to_square` (0-63 indexing).
Pawn promotion is automatically handled (promotes to Queen).

### Reward Function

| Event | Reward | Description |
|-------|--------|-------------|
| **Win** | `+1.0` | Checkmate or Opponent Timeout |
| **Loss** | `-1.0` | Checkmated or Agent Timeout |
| **Draw** | `0.0` | Stalemate, repetition, insufficient material |
| **Illegal**| `-10.0`| Attempting a pseudo-legal or invalid move. |

In **Self-Play**, the reward is always relative to the agent who **just moved**. If Black moves and Checkmates White, the reward returned is `+1.0` (Black Wins).

### Assets & Credits

The chess piece images used for visual observations are created by **Colin M.L. Burnett**.
- **Source:** Wikimedia Commons
- **License:** [Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)

---

## License

This project uses `python-chess` (GPL-3.0) and is therefore released under the **GNU General Public License v3.0**.
