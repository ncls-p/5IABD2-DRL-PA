import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.ppo import PPOAgent
from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe


def plot_metrics(data, title, env_name, window_size=100):
    plt.plot(data, color="gray", alpha=0.3, label="Raw")

    # Calculate moving average
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    plt.plot(
        range(window_size - 1, len(data)),
        moving_avg,
        color="red",
        label=f"{window_size}-Episode Moving Average",
    )

    # Add milestone markers
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


def print_scores_at_milestones(scores, env_name):
    milestones = [1000, 10000, 100000, 1000000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = np.mean(scores[max(0, milestone - 100) : milestone])
            print(f"Average score at episode {milestone}: {avg_score:.2f}")


def run_ppo_example(
    env_class,
    env_name,
    num_episodes=10000,
    learning_rate=0.001,
    gamma=0.99,
    clip_epsilon=0.2,
    epochs=10,
    batch_size=64,
):
    env = env_class()

    agent = PPOAgent(
        env=env,
        lr=learning_rate,
        gamma=gamma,
        clip_epsilon=clip_epsilon,
        epochs=epochs,
        batch_size=batch_size,
        device="cpu",
    )

    print(f"\nStarting training on {env_name}...")
    print(
        f"Configuration: {num_episodes=}, {learning_rate=}, {gamma=}, {clip_epsilon=}, {epochs=}, {batch_size=}\n"
    )

    scores, steps_per_episode, action_times = agent.train(num_episodes=num_episodes)

    print(f"\nTraining completed on {env_name}")
    print("Final Statistics:")
    print(f"Average Score (last 100): {np.mean(scores[-100:]):.2f}")
    print(f"Average Steps (last 100): {np.mean(steps_per_episode[-100:]):.2f}")
    print(f"Average Action Time (last 100): {np.mean(action_times[-100:]):.4f}s\n")

    # Create plots
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 1, 1)
    plot_metrics(scores, "Scores", env_name)

    plt.subplot(3, 1, 2)
    plot_metrics(steps_per_episode, "Steps per Episode", env_name)

    plt.subplot(3, 1, 3)
    plot_metrics(action_times, "Action Time (seconds)", env_name)

    plt.tight_layout()
    plt.savefig(
        f"src/metrics/plot/ppo/ppo_{env_name.lower().replace(' ', '_')}_metrics.png"
    )
    plt.close()

    print_scores_at_milestones(scores, env_name)

    return scores, steps_per_episode, action_times


def main():
    os.makedirs("src/metrics/plot/ppo", exist_ok=True)

    # Environment-specific configurations for PPO
    env_configs_ppo = {
        # (LineWorld, "Line World"): {
        #     "num_episodes": 100000,
        #     "learning_rate": 0.001,
        #     "gamma": 0.99,
        #     "clip_epsilon": 0.2,
        #     "epochs": 10,
        #     "batch_size": 64,
        # },
        # (GridWorld, "Grid World"): {
        #     "num_episodes": 100000,
        #     "learning_rate": 0.001,
        #     "gamma": 0.99,
        #     "clip_epsilon": 0.2,
        #     "epochs": 10,
        #     "batch_size": 64,
        # },
        # (TicTacToe, "Tic Tac Toe"): {
        #     "num_episodes": 100000,
        #     "learning_rate": 0.001,
        #     "gamma": 0.99,
        #     "clip_epsilon": 0.2,
        #     "epochs": 10,
        #     "batch_size": 128,
        # },
        (Farkle, "Farkle"): {
            "num_episodes": 10000,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "epochs": 10,
            "batch_size": 64,
        },
    }

    for (env_class, env_name), config in env_configs_ppo.items():
        print(f"\nRunning PPO on {env_name}...")
        run_ppo_example(env_class, env_name, **config)


if __name__ == "__main__":
    main()
