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

### Basic Usage

```python
import gymnasium as gym
import bullet_chess_env  # registers the environment

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

### Self-Play Mode

You can enable self-play mode to control both White and Black pieces. This disables the automatic random opponent.

```python
import gymnasium as gym
import bullet_chess_env

# Enable self-play at initialization
env = gym.make("BulletChess-v0", self_play=True)

# OR enable via reset options
# env = gym.make("BulletChess-v0")
# env.reset(options={"self_play": True})

obs, info = env.reset()

# Play a move for White
obs, reward, terminated, truncated, info = env.step(white_action)

if not terminated:
    # Play a move for Black
    obs, reward, terminated, truncated, info = env.step(black_action)
```

### Real-Time Constraints

To properly simulate bullet chess, you should use the `RealTimeClock` wrapper. This wrapper measures the time your agent takes to compute an action and deducts it from the in-game clock.

```python
import gymnasium as gym
import bullet_chess_env
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

The observation is a `Dict` space containing the board representation and the game state (including time).

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `board` | `(8, 8, 12)` | `float32` | 8x8 spatial representation of the board. <br>Layers 0-5: White P, N, B, R, Q, K <br>Layers 6-11: Black P, N, B, R, Q, K |
| `state` | `(8,)` | `float32` | Global state vector containing flags and time info. |

**State Vector Layout:**
- Index 0: **Turn** (1.0 = White, 0.0 = Black)
- Index 1: **White Kingside Castle** (1.0 if available)
- Index 2: **White Queenside Castle** (1.0 if available)
- Index 3: **Black Kingside Castle** (1.0 if available)
- Index 4: **Black Queenside Castle** (1.0 if available)
- Index 5: **En Passant** (1.0 if available)
- Index 6: **White Time** (Normalized 0.0-1.0, where 1.0 = 60s)
- Index 7: **Black Time** (Normalized 0.0-1.0, where 1.0 = 60s)

### Action Space

The action space is `Discrete(4096)`. 

Each integer action represents a move from one square to another:
`action = from_square * 64 + to_square`

- Squares are indexed 0-63 (A1=0, B1=1, ... H8=63).
- **Pawn Promotion**: If a pawn moves to the last rank, it is automatically promoted to a **Queen**. Under-promotions (Knight, Rook, Bishop) are not currently supported in the action space.

### Reward Function

| Event | Reward | Description |
|-------|--------|-------------|
| **Win** | `+1.0` | Checkmate or Opponent Timeout |
| **Loss** | `-1.0` | Checkmated or Agent Timeout |
| **Draw** | `0.0` | Stalemate, repetition, insufficient material |
| **Illegal Move** | `-10.0` | Attempting a pseudo-legal or invalid move. Episode terminates immediately. |

### Opponent
The environment includes a built-in "Random Opponent" (default).
- **Moves**: Chooses a random legal move.
- **Time**: Consumes a random amount of time (0.1s to 0.5s) per move to simulate a human processing delay.

To disable the opponent and control both sides manually, see **Self-Play Mode** above.

---

## License

This project uses `python-chess` (GPL-3.0) and is therefore released under the **GNU General Public License v3.0**.
