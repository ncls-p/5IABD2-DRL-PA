import os
import sys
import numpy as np

from matplotlib import pyplot as plt


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.d_dqn import DoubleDQNAgent
from src.environments.line_world import LineWorld
from src.environments.grid_world import GridWorld
from src.environments.farkle import Farkle
from src.environments.tic_tac_toe import TicTacToe


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def run_ddqn_example(env_class, env_name, num_episodes=10000):
    env = env_class()
    state_size = len(env.state_vector())
    action_size = env.num_actions()
    learning_rate = 0.0005 * 2
    gamma = 0.99
    target_update_freq = 10

    if action_size <= 0:
        raise ValueError(f"Invalid action_size: {action_size}")

    agent = DoubleDQNAgent(
        env,
        state_size=state_size,
        action_size=action_size,
        gamma=gamma,
        lr=learning_rate,
        target_update_freq=target_update_freq,
    )

    scores, steps_per_episode, action_times = agent.train(num_episodes=num_episodes)

    return scores, steps_per_episode, action_times


def print_scores_at_milestones(scores, env_name):
    milestones = [1000, 10000, 100000, 1000000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = sum(scores[milestone - 100 : milestone]) / 100
            print(f"At {milestone} episodes: {avg_score:.2f}")


def plot_metrics(data, metric_name, env_name):
    moving_avg = moving_average(data, window_size=100)
    plt.plot(data, color="gray", alpha=0.2, label=f"Raw {metric_name}")
    plt.plot(
        range(len(moving_avg)), moving_avg, color="red", label=f"Average {metric_name}"
    )

    milestones = [1000, 10000, 100000, 1000000]
    for milestone in milestones:
        if milestone <= len(data):
            avg_value = sum(data[milestone - 100 : milestone]) / 100
            plt.axvline(x=milestone, color="green", linestyle="--", alpha=0.7)
            plt.text(
                milestone,
                plt.ylim()[1],
                f"{milestone}\n{avg_value:.2f}",
                horizontalalignment="center",
                verticalalignment="top",
            )

    plt.title(f"{metric_name} - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)


def main():
    for env_class, env_name in [
        # (LineWorld, "Line World"),
        # (GridWorld, "Grid World"),
        # (TicTacToe, "Tic Tac Toe"),
        (Farkle, "Farkle"),
    ]:
        scores, steps_per_episode, action_times = run_ddqn_example(env_class, env_name)

        plt.figure(figsize=(15, 12))

        plt.subplot(3, 1, 1)
        plot_metrics(scores, "Scores", env_name)

        plt.subplot(3, 1, 2)
        plot_metrics(steps_per_episode, "Steps per Episode", env_name)

        plt.subplot(3, 1, 3)
        plot_metrics(action_times, "Action Time (seconds)", env_name)

        plt.tight_layout()

        plt.savefig(
            f"src/metrics/plot/d_dqn/ddqn_{env_name.lower().replace(' ', '_')}_metrics.png"
        )
        plt.close()

        print_scores_at_milestones(scores, env_name)


if __name__ == "__main__":
    main()
