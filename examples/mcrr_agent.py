import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.mcrr import RandomRolloutAgent
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

    plt.title(f"{title} - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()


def run_mcrr_example(
    env_class,
    env_name,
    num_rollouts=100,
    num_episodes=100000,
    gamma=0.99,
):
    env = env_class()
    agent = RandomRolloutAgent(
        env,
        num_rollouts=num_rollouts,
        gamma=gamma,
        env_name=env_name
    )

    scores, steps_per_episode, action_times = agent.train(
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
        f"src/metrics/plot/mcrr/mcrr_{env_name.lower().replace(' ', '_')}_metrics.png"
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
    os.makedirs("src/metrics/plot/mcrr", exist_ok=True)

    env_configs = {
        # (LineWorld, "Line World"): {
        #     "num_rollouts": 100,
        #     "num_episodes": 100000,
        #     "gamma": 0.99,
        # },
        # (GridWorld, "Grid World"): {
        #     "num_rollouts": 200,
        #     "num_episodes": 100000,
        #     "gamma": 0.99,
        # },
        # (TicTacToe, "Tic Tac Toe"): {
        #     "num_rollouts": 500,
        #     "num_episodes": 100000,
        #     "gamma": 0.95,
        # },
        (Farkle, "Farkle"): {
            "num_rollouts": 1000,
            "num_episodes": 10000,
            "gamma": 0.95,
        },
    }

    for (env_class, env_name), config in env_configs.items():
        print(f"\nRunning Monte Carlo Random Rollout on {env_name}...")
        scores, steps, times = run_mcrr_example(
            env_class,
            env_name,
            **config
        )


if __name__ == "__main__":
    main()
