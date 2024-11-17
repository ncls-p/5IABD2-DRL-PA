import os
import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.tab_q_learning import TabularQLearningAgent
from src.environments.line_world import LineWorld
from src.environments.grid_world import GridWorld
from src.environments.farkle import Farkle
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
    milestones = [100, 500, 1000, 5000, 10000]
    for milestone in milestones:
        if milestone <= len(data):
            avg_value = np.mean(data[max(0, milestone - 100):milestone])
            plt.axvline(x=milestone, color='green', linestyle='--', alpha=0.3)
            plt.plot(milestone, avg_value, 'go', alpha=0.6)
            plt.annotate(
                f'Milestone {milestone}\nAvg: {avg_value:.2f}',
                xy=(milestone, avg_value),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    plt.title(f"{title} - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()


def run_tab_q_learn_example(env_class, env_name, num_episodes=10000):
    env = env_class()
    learning_rate = 0.0005 * 2
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    # Initialize agent
    agent = TabularQLearningAgent(
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    # Train the agent
    scores, steps_per_episode, action_times = agent.train(num_episodes=num_episodes)

    # Create a figure with three subplots
    plt.figure(figsize=(15, 12))

    # Plot scores
    plt.subplot(3, 1, 1)
    plot_metrics(scores, "Scores", env_name)

    # Plot steps per episode
    plt.subplot(3, 1, 2)
    plot_metrics(steps_per_episode, "Steps per Episode", env_name)

    # Plot action times
    plt.subplot(3, 1, 3)
    plot_metrics(action_times, "Action Time (seconds)", env_name)

    plt.tight_layout()
    plt.savefig(
        f"src/metrics/plot/tab_q_learn/tab_q_learn_{env_name.lower().replace(' ', '_')}_metrics.png"
    )
    plt.close()

    # Print score statistics at milestones
    print_scores_at_milestones(scores, env_name)

    return scores, steps_per_episode, action_times


def print_scores_at_milestones(scores, env_name):
    milestones = [100, 500, 1000, 5000, 10000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = np.mean(scores[max(0, milestone - 100):milestone])
            print(f"Average score at episode {milestone}: {avg_score:.2f}")


def main():
    # Create metrics directory if it doesn't exist
    os.makedirs("src/metrics/plot/tab_q_learn", exist_ok=True)

    # Run Q-learning on each environment
    for env_class, env_name in [
        (LineWorld, "Line World"),
        (GridWorld, "Grid World"),
        (TicTacToe, "Tic Tac Toe"),
        (Farkle, "Farkle"),
    ]:
        print(f"\nRunning Tabular Q-Learning on {env_name}...")
        scores, steps, times = run_tab_q_learn_example(env_class, env_name)


if __name__ == "__main__":
    main()
