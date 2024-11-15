import os
import sys
import time
import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.deep_q_learning import DQNAgent
from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def run_dqn_example(env_class, env_name, num_episodes=10001):
    env = env_class()
    state_size = len(env.state_vector())
    action_size = env.num_actions()
    buffer_size = 10000
    learning_rate = 0.001
    gamma = 0.99
    batch_size = 16
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    agent = DQNAgent(
        env,
        state_size=state_size,
        action_size=action_size,
        buffer_size=buffer_size,
        lr=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
    )

    scores = []
    steps_per_episode = []
    action_times = []
    epsilon = epsilon_start

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = env.state_vector()
        total_reward = 0
        done = False
        episode_steps = 0
        episode_action_times = []

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Measure action execution time
            action_start = time.time()
            action = agent.choose_action(state_tensor, epsilon)
            action_time = time.time() - action_start
            episode_action_times.append(action_time)

            next_state, reward, done, _ = env.step(action)
            next_state = env.state_vector()
            agent.memory.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            episode_steps += 1

            if len(agent.memory) > batch_size:
                agent.learn(agent.memory.sample())

        scores.append(total_reward)
        steps_per_episode.append(episode_steps)
        action_times.append(np.mean(episode_action_times))
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            avg_action_time = np.mean(action_times[-100:])
            print(
                f"Episode {episode}/{num_episodes}, "
                f"Avg Score: {avg_score:.5f}, "
                f"Avg Steps: {avg_steps:.2f}, "
                f"Avg Action Time: {avg_action_time:.5f}s, "
                f"Epsilon: {epsilon:.2f}"
            )

        if episode % 100 == 0:
            agent.update_target_network()

    # Plot scores
    plt.figure(figsize=(15, 7))
    plt.subplot(3, 1, 1)
    plot_metrics(scores, "Scores", env_name)

    # Plot steps per episode
    plt.subplot(3, 1, 2)
    plot_metrics(steps_per_episode, "Steps per Episode", env_name)

    # Plot action execution times
    plt.subplot(3, 1, 3)
    plot_metrics(action_times, "Average Action Time (seconds)", env_name)

    plt.tight_layout()
    plt.savefig(
        f"src/metrics/plot/dqn/dqn_{env_name.lower().replace(' ', '_')}_metrics.png"
    )
    plt.close()

    return scores, steps_per_episode, action_times


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
        # (Farkle, "Farkle"),
    ]:
        scores, steps_per_episode, action_times = run_dqn_example(env_class, env_name)
        print_scores_at_milestones(scores, env_name)


if __name__ == "__main__":
    main()
