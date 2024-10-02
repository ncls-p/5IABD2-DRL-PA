import os
import sys
import time
from typing import Type

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.farkle import Farkle
from src.environments.grid_world import GridWorld
from src.environments.line_world import LineWorld
from src.environments.tic_tac_toe import TicTacToe
from src.metrics.performance_metrics import PerformanceMetrics, calculate_metrics


class RandomAgent:
    def choose_action(self, available_actions: np.ndarray) -> int:
        return np.random.choice(available_actions)


def play_episodes(
    env, agent1, agent2, num_episodes: int = 100000
) -> PerformanceMetrics:
    metrics = PerformanceMetrics()
    start_time = time.time()

    for _ in range(num_episodes):
        env.reset()
        done = False
        episode_length = 0

        # Randomize which agent plays first
        if isinstance(env, Farkle):
            current_agent = agent1 if np.random.rand() < 0.5 else agent2
            env.current_player = 1 if current_agent == agent1 else 2
        elif isinstance(env, TicTacToe):
            current_agent = agent1  # For TicTacToe, alternate moves between agents
            env.current_player = 1
        else:
            current_agent = agent1

        while not done:
            action = current_agent.choose_action(env.available_actions())
            _, reward, done, info = env.step(action)
            episode_length += 1

            if isinstance(env, Farkle):
                if info.get("turn_ended", False):
                    current_agent = agent2 if current_agent == agent1 else agent1
            elif isinstance(env, TicTacToe):
                current_agent = agent2 if current_agent == agent1 else agent1
            else:
                pass

        if isinstance(env, Farkle):
            total_reward = env.scores[0] - env.scores[1]
        elif isinstance(env, TicTacToe):
            if env.winner == 1:
                total_reward = 1  # Agent1 wins
            elif env.winner == 2:
                total_reward = -1  # Agent2 wins
            else:
                total_reward = 0  # Draw
        else:
            total_reward = reward

        metrics.add_episode(total_reward, episode_length)

    end_time = time.time()
    total_time = end_time - start_time
    episodes_per_second = num_episodes / total_time
    metrics.set_episodes_per_second(episodes_per_second)

    return metrics


def run_random_agent(
    env_class: Type[LineWorld | GridWorld | TicTacToe | Farkle], env_name: str
):
    env = env_class()
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    num_episodes = 1000  # Define the number of episodes

    print(f"\nTesting Random Agents on {env_name}")
    print("=" * 40)

    metrics = play_episodes(env, agent1, agent2, num_episodes=num_episodes)

    print(f"Average Score: {metrics.get_average_score():.2f}")
    print(f"Average Episode Length: {metrics.get_average_length():.2f}")
    print(f"Episodes per second: {metrics.get_episodes_per_second():.2f}")

    episode_rewards = metrics.scores
    calculated_metrics = calculate_metrics(episode_rewards)

    print(f"Standard Deviation: {calculated_metrics['std_dev']:.2f}")
    print(f"Min Reward: {calculated_metrics['min_reward']:.2f}")
    print(f"Max Reward: {calculated_metrics['max_reward']:.2f}")
    print(f"Median Reward: {calculated_metrics['median_reward']:.2f}")

    if isinstance(env, Farkle):
        print("\nFarkle-specific statistics:")
        agent1_wins = np.sum(np.array(metrics.scores) > 0)
        agent2_wins = np.sum(np.array(metrics.scores) < 0)
        ties = np.sum(np.array(metrics.scores) == 0)
        print(f"Agent1 win rate: {agent1_wins / num_episodes:.2%}")
        print(f"Agent2 win rate: {agent2_wins / num_episodes:.2%}")
        print(f"Tie rate: {ties / num_episodes:.2%}")
        print(f"Average turns per game: {metrics.get_average_length() / 2:.2f}")
    elif isinstance(env, TicTacToe):
        print("\nTicTacToe-specific statistics:")
        agent1_wins = np.sum(np.array(metrics.scores) > 0)
        agent2_wins = np.sum(np.array(metrics.scores) < 0)
        ties = np.sum(np.array(metrics.scores) == 0)
        print(f"Agent1 win rate: {agent1_wins / num_episodes:.2%}")
        print(f"Agent2 win rate: {agent2_wins / num_episodes:.2%}")
        print(f"Tie rate: {ties / num_episodes:.2%}")
    else:
        success_rate = np.mean([1 for s in metrics.scores if s > 0])
        print(f"Success Rate: {success_rate:.2%}")


if __name__ == "__main__":
    for env_class, env_name in [
        (LineWorld, "LineWorld"),
        (GridWorld, "GridWorld"),
        (TicTacToe, "TicTacToe"),
        (Farkle, "Farkle"),
    ]:
        run_random_agent(env_class, env_name)