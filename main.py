import gymnasium as gym
import time
import sys

# Ensure gym_bullet_chess is importable
# If you have installed the package via 'pip install -e .', this import works anywhere.
try:
    import gym_bullet_chess
    from gym_bullet_chess.wrappers import RealTimeClock
except ImportError:
    print("Error: gym_bullet_chess not found.")
    print("Please ensure you have installed the package using 'pip install -e .'")
    sys.exit(1)


def main():
    print("Initializing BulletChess-v0...")

    # 1. Initialize the environment
    # We use render_mode="ansi" to print the board to the terminal
    env = gym.make("BulletChess-v0", render_mode="ansi")

    # 2. Wrap it with RealTimeClock
    # This wrapper captures the time taken by 'env.step' calls (and overhead)
    # and deducts it from the agent's game clock.
    env = RealTimeClock(env)

    # 3. Reset the environment
    obs, info = env.reset()

    # Helper to decode time from state vector (indices 6 and 7)
    # State indices: 6=WhiteTime, 7=BlackTime (Normalized 0-1, where 1=60s)
    initial_white_time = obs["state"][6] * 60.0
    print(f"Game Started. Initial White Time: {initial_white_time:.1f}s")

    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1

        # --- AGENT LOGIC ---
        # Simulate some "thinking" time (e.g., 0.05s)
        # In a real scenario, this would be your model inference time.
        time.sleep(0.05)

        # For this demo, we pick a random action
        action = env.action_space.sample()
        # -------------------

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

        white_time = obs["state"][6] * 60.0
        black_time = obs["state"][7] * 60.0

        # Print progress
        print(
            f"Move {step_count}: Reward={reward} | W_Time={white_time:.2f}s | B_Time={black_time:.2f}s"
        )

        # Render board occasionally
        if step_count % 5 == 0:
            print("\n" + env.render())
            print("-" * 20)

    print(f"\nGame Over!")
    print(f"Total Steps: {step_count}")
    print(f"Final Reward: {total_reward}")
    print(f"Info: {info}")

    env.close()


if __name__ == "__main__":
    main()
