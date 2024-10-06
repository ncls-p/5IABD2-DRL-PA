import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mcts import MCTSAgent
from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld


def run_mcts_example(env_class, env_name, num_simulations=100, num_episodes=1000):
    print(f"\nRunning MCTS on {env_name}")
    env = env_class()
    agent = MCTSAgent(env, num_simulations=num_simulations)
    scores = agent.train(num_episodes=num_episodes)

    # Plot the learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title(f"MCTS Learning Curve - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig(f"mcts_{env_name.lower().replace(' ', '_')}_learning_curve.png")
    plt.close()

    return scores


def main():
    # Run MCTS on Farkle
    farkle_scores = run_mcts_example(
        Farkle, "Farkle", num_simulations=100, num_episodes=1000
    )

    # Run MCTS on LineWorld
    line_world_scores = run_mcts_example(
        LineWorld, "Line World", num_simulations=100, num_episodes=1000
    )

    # Run MCTS on GridWorld
    grid_world_scores = run_mcts_example(
        GridWorld, "Grid World", num_simulations=100, num_episodes=1000
    )

    # Print final average scores
    print("\nFinal Average Scores:")
    print(f"Farkle: {sum(farkle_scores[-100:]) / 100:.2f}")
    print(f"Line World: {sum(line_world_scores[-100:]) / 100:.2f}")
    print(f"Grid World: {sum(grid_world_scores[-100:]) / 100:.2f}")


if __name__ == "__main__":
    main()
