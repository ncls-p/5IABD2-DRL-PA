import os
import sys
from typing import Type

import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe
from src.metrics.performance_metrics import PerformanceMetrics, calculate_metrics


class RandomAgent:
    def choose_action(self, available_actions: np.ndarray) -> int:
        return np.random.choice(available_actions)


def play_episodes(env, agent, num_episodes: int = 1000) -> PerformanceMetrics:
    metrics = PerformanceMetrics()
    start_time = time.time()

    for _ in range(num_episodes):
        env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action = agent.choose_action(env.available_actions())
            _, reward, done, _ = env.step(action)
            total_reward += reward
            episode_length += 1

        metrics.add_episode(total_reward, episode_length)

    end_time = time.time()
    total_time = end_time - start_time
    episodes_per_second = num_episodes / total_time
    metrics.set_episodes_per_second(episodes_per_second)

    return metrics


def run_random_agent(env_class: Type[LineWorld | GridWorld | TicTacToe], env_name: str):
    env = env_class()
    agent = RandomAgent()

    print(f"\nTesting Random Agent on {env_name}")
    print("=" * 40)

    metrics = play_episodes(env, agent)

    print(f"Average Score: {metrics.get_average_score():.2f}")
    print(f"Average Episode Length: {metrics.get_average_length():.2f}")
    print(f"Episodes per second: {metrics.get_episodes_per_second():.2f}")

    episode_rewards = metrics.scores
    calculated_metrics = calculate_metrics(episode_rewards)

    print(f"Standard Deviation: {calculated_metrics['std_dev']:.2f}")
    print(f"Min Reward: {calculated_metrics['min_reward']:.2f}")
    print(f"Max Reward: {calculated_metrics['max_reward']:.2f}")
    print(f"Median Reward: {calculated_metrics['median_reward']:.2f}")
    print(f"Success Rate: {calculated_metrics['success_rate']:.2%}")


if __name__ == "__main__":
    for env_class, env_name in [
        (LineWorld, "LineWorld"),
        (GridWorld, "GridWorld"),
        (TicTacToe, "TicTacToe"),
    ]:
        run_random_agent(env_class, env_name)
