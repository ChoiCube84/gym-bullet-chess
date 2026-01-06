# gym-bullet-chess

A Gymnasium-compatible bullet chess environment for reinforcement learning research under real-time decision constraints.

This environment models bullet-style chess, where agents must trade off move quality against limited decision time.

The environment is implemented using python-chess and follows the Gymnasium API.

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

Example usage:

```python
import gymnasium as gym
import bullet_chess_env  # registers the environment

env = gym.make("BulletChess-v0")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

---

## Environment

- ID: BulletChess-v0
- Game: Chess (bullet time control)
- Players: 2
- Action space: Chess moves
- Observation space: Board state and clock information

---

## License

This project uses python-chess (GPL-3.0) and is therefore released under the GNU General Public License v3.0.
