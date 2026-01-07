from gymnasium.envs.registration import register

# Register the environment
register(
    id="BulletChess-v0",
    entry_point="gym_bullet_chess.envs:BulletChessEnv",
    max_episode_steps=300,
)

# Convenience imports
from gym_bullet_chess.wrappers import RealTimeClock
