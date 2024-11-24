import os
import sys
import numpy as np
import torch
import time
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.reinforce import REINFORCEAgent
from src.environments.line_world import LineWorld
from src.environments.grid_world import GridWorld
from src.environments.tic_tac_toe import TicTacToe
from src.environments.farkle import Farkle


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


def run_reinforce_example(
    env_class,
    env_name,
    num_episodes=10000,
    learning_rate=0.001,
    gamma=0.99,
    hidden_size=64
):
    env = env_class()
    state_size = len(env.state_vector())
    action_size = env.num_actions()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = REINFORCEAgent(
        env=env,
        state_size=state_size,
        action_size=action_size,
        lr=learning_rate,
        gamma=gamma,
        device=device
    )

    scores = []
    steps_per_episode = []
    action_times = []

    for episode in range(num_episodes):
        state = env.reset()
        state = env.state_vector()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_action_times = []

        states = []
        actions = []
        rewards = []
        log_probs = []

        while not done:
            start_time = time.time()
            action, log_prob = agent.choose_action(state)
            action_time = time.time() - start_time
            episode_action_times.append(action_time)

            next_state, reward, done, _ = env.step(action)
            next_state = env.state_vector()

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            episode_reward += reward
            episode_steps += 1

        agent.train_episode(rewards, log_probs)

        scores.append(episode_reward)
        steps_per_episode.append(episode_steps)
        action_times.append(np.mean(episode_action_times))

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            avg_action_time = np.mean(action_times[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Score: {avg_score:.2f}, "
                f"Avg Steps: {avg_steps:.2f}, "
                f"Avg Action Time: {avg_action_time:.5f}s"
            )

        if episode > 0 and episode % 5000 == 0:
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'scores': scores,
            }, f'reinforce_checkpoint_{env_name}_{episode}.pt')

    return scores, steps_per_episode, action_times


def print_scores_at_milestones(scores, env_name):
    milestones = [1000, 5000, 10000]
    print(f"\n{env_name} Scores at Milestones:")
    for milestone in milestones:
        if milestone <= len(scores):
            avg_score = np.mean(scores[max(0, milestone - 100):milestone])
            print(f"Average score at episode {milestone}: {avg_score:.2f}")


def main():
    os.makedirs("src/metrics/plot/reinforce", exist_ok=True)

    env_configs = {
        # (LineWorld, "Line World"): {
        #     "num_episodes": 10000,
        #     "learning_rate": 0.001,
        #     "gamma": 0.99,
        # },
        (GridWorld, "Grid World"): {
            "num_episodes": 10000,
            "learning_rate": 0.00001,
            "gamma": 0.99,
        }
        # (TicTacToe, "Tic Tac Toe"): {
        #     "num_episodes": 10000,
        #     "learning_rate": 0.0005,
        #     "gamma": 0.95,
        # }
        # (Farkle, "Farkle"): {
        #     "num_episodes": 1000,
        #     "learning_rate": 0.0005,
        #     "gamma": 0.95,
        # }
    }

    for (env_class, env_name), config in env_configs.items():
        print(f"\nRunning REINFORCE on {env_name}...")
        scores, steps, times = run_reinforce_example(
            env_class,
            env_name,
            **config
        )

        plt.figure(figsize=(15, 12))

        plt.subplot(3, 1, 1)
        plot_metrics(scores, "Scores", env_name)

        plt.subplot(3, 1, 2)
        plot_metrics(steps, "Steps per Episode", env_name)

        plt.subplot(3, 1, 3)
        plot_metrics(times, "Action Time (seconds)", env_name)

        plt.tight_layout()
        plt.savefig(
            f"src/metrics/plot/reinforce/reinforce_{env_name.lower().replace(' ', '_')}_metrics.png"
        )
        plt.close()

        print_scores_at_milestones(scores, env_name)


if __name__ == "__main__":
    main()
