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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

def run_tab_q_learn_example(env_class, env_name, num_episodes=100000):
    env = env_class()
    learning_rate=0.0005*2
    gamma=0.99
    epsilon_start=1.0
    epsilon_min=1.0
    epsilon_decay=0.95

    # Initialize agent
    agent = TabularQLearningAgent(env=env, learning_rate=learning_rate, gamma=gamma, epsilon_start=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)

    # Train the agent
    scores = agent.train(num_episodes=num_episodes)

    moving_avg = moving_average(scores, window_size=100)

    plt.figure(figsize=(15, 7))
    plt.plot(scores, color="gray", alpha=0.2, label="Raw Scores")
    plt.plot(range(len(moving_avg)), moving_avg, color="red", label="Average Scores")

    milestones = [1000, 10000, 100000, 1000000]
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = sum(scores[milestone - 100 : milestone]) / 100
            plt.axvline(x=milestone, color="green", linestyle="--", alpha=0.7)
            plt.text(
                milestone,
                plt.ylim()[1],
                f"{milestone}\n{avg_score:.2f}",
                horizontalalignment="center",
                verticalalignment="top",
            )

    plt.title(f"tab q learning Learning Curve - {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    plt.savefig(f"src/metrics/plot/tab_q_learn/tab_q_learn_{env_name.lower().replace(' ', '_')}_learning_curve.png")
    plt.close()

    return scores

def print_scores_at_milestones(scores, env_name):
    milestones = [100, 500, 1000, 5000, 10000, 100000, 1000000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = sum(scores[milestone - 100 : milestone]) / 100
            print(f"At {milestone} episodes: {avg_score:.2f}")

def main():
    for env_class, env_name in [
        (LineWorld, "Line World"),
        (GridWorld, "Grid World"),
        (TicTacToe, "Tic Tac Toe"),
        (Farkle, "Farkle"),
    ]:
        scores = run_tab_q_learn_example(env_class, env_name)
        print_scores_at_milestones(scores, env_name)

if __name__ == "__main__":
    main()

