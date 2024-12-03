import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mcts import MCTSAgent
from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe


def plot_metrics(data, title, env_name, window_size=100):
    plt.plot(data, color="gray", alpha=0.3, label="Raw")

    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    plt.plot(
        range(window_size - 1, len(data)),
        moving_avg,
        color="red",
        label=f"{window_size}-Episode Moving Average",
    )

    milestones = [1000, 10000, 100000, 1000000]
    for milestone in milestones:
        if milestone <= len(data):
            avg_value = np.mean(data[max(0, milestone - 100) : milestone])
            plt.axvline(x=milestone, color="green", linestyle="--", alpha=0.3)
            plt.plot(milestone, avg_value, "go", alpha=0.6)
            plt.annotate(
                f"Milestone {milestone}\nAvg: {avg_value:.2f}",
                xy=(milestone, avg_value),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

    plt.title(f"{title} - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()


def run_mcts_example(
    env_class,
    env_name,
    num_simulations=100,
    num_episodes=10000,
    exploration_weight=1.414,
    simulation_temp=0.5,
):
    env = env_class()
    agent = MCTSAgent(
        env,
        num_simulations=num_simulations,
        exploration_weight=exploration_weight,
        simulation_temp=simulation_temp,
        max_depth=50,
    )

    scores, steps_per_episode, epsilon_values, action_times = agent.train(
        num_episodes=num_episodes
    )

    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plot_metrics(scores, "Scores", env_name)

    plt.subplot(3, 1, 2)
    plot_metrics(steps_per_episode, "Steps per Episode", env_name)

    plt.subplot(3, 1, 3)
    plot_metrics(action_times, "Action Time (seconds)", env_name)

    plt.tight_layout()
    plt.savefig(
        f"src/metrics/plot/mcts/mcts_{env_name.lower().replace(' ', '_')}_metrics.png"
    )
    plt.close()

    print_scores_at_milestones(scores, env_name)

    return scores, steps_per_episode, action_times


def print_scores_at_milestones(scores, env_name):
    milestones = [1000, 10000, 100000, 1000000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = np.mean(scores[max(0, milestone - 100) : milestone])
            print(f"Average score at episode {milestone}: {avg_score:.2f}")


def main():
    os.makedirs("src/metrics/plot/mcts", exist_ok=True)

    # Environment-specific configurations
    env_configs = {
        # (LineWorld, "Line World"): {
        #     "num_simulations": 100,
        #     "num_episodes": 100000,
        #     "exploration_weight": 1.414,
        # },
        # (GridWorld, "Grid World"): {
        #     "num_simulations": 1000,
        #     "num_episodes": 100000,
        #     "exploration_weight": 4,
        # },
        # (TicTacToe, "Tic Tac Toe"): {
        #     "num_simulations": 2000,
        #     "num_episodes": 100000,
        #     "exploration_weight": 6,
        # },
        (Farkle, "Farkle"): {
            "num_simulations": 3000,
            "num_episodes": 10000,
            "exploration_weight": 8,
        },
    }

    for (env_class, env_name), config in env_configs.items():
        print(f"\nRunning MCTS on {env_name}...")
        scores, steps, times = run_mcts_example(
            env_class,
            env_name,
            num_simulations=config["num_simulations"],
            num_episodes=config["num_episodes"],
            exploration_weight=config.get("exploration_weight", 1.414),
            simulation_temp=config.get("simulation_temp", 0.5),
        )


if __name__ == "__main__":
    main()
