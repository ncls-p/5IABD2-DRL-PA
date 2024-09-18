import os
import sys
from typing import Type
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe
from src.metrics.performance_metrics import PerformanceMetrics


class PlayerAgent:
    def choose_action(self, available_actions: np.ndarray) -> int:
        while True:
            action = input("Enter your action: ")
            try:
                action = int(action)
                if action in available_actions:
                    return action
                else:
                    print("Invalid action. Please choose from available actions.")
            except ValueError:
                print("Please enter a valid number.")


# Use a dictionary for environment instructions to avoid if-elif chains
ENVIRONMENT_INSTRUCTIONS = {
    "LineWorld": "Move left (0) or right (1) to reach the goal.",
    "GridWorld": "Move up (0), right (1), down (2), or left (3) to reach the goal.",
    "TicTacToe": "Enter a number from 0 to 8 to place your mark (X) in the corresponding cell.\n"
    "The board is numbered from left to right, top to bottom.",
}


def play_game(env_class: Type[LineWorld | GridWorld | TicTacToe], env_name: str):
    env = env_class()
    agent = PlayerAgent()
    metrics = PerformanceMetrics()

    print(f"\nPlaying {env_name}")
    print("=" * 40)

    env.reset()
    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        print(f"\nCurrent state:\n{env.render()}")
        print(f"Available actions: {env.available_actions()}")

        action = agent.choose_action(env.available_actions())
        _, reward, done, _ = env.step(action)

        total_reward += reward
        episode_length += 1

        print(f"Reward: {reward}")

    print(f"\nGame Over! Final state:\n{env.render()}")
    print(f"Total reward: {total_reward}")
    print(f"Episode length: {episode_length}")

    metrics.add_episode(total_reward, episode_length)
    return metrics


def main():
    environments = {
        "1": (LineWorld, "LineWorld"),
        "2": (GridWorld, "GridWorld"),
        "3": (TicTacToe, "TicTacToe"),
    }

    print("Choose an environment to play:")
    for key, (_, name) in environments.items():
        print(f"{key}. {name}")

    choice = input("Enter your choice (1-3): ")

    if choice in environments:
        env_class, env_name = environments[choice]
        print("\nInstructions:")
        print(
            ENVIRONMENT_INSTRUCTIONS.get(
                env_name, "No specific instructions available."
            )
        )
        metrics = play_game(env_class, env_name)
    else:
        print("Invalid choice. Exiting.")
        return

    print("\nGame Statistics:")
    print(f"Score: {metrics.get_average_score()}")
    print(f"Episode Length: {metrics.get_average_length()}")


if __name__ == "__main__":
    main()
