"""
Gym Bullet Chess Environment Package.
Exports the BulletChessEnv class and registers the gymnasium environment.
"""

from gymnasium.envs.registration import register

register(
    id="BulletChess-v0",
    entry_point="gym_bullet_chess.envs:BulletChessEnv",
)
