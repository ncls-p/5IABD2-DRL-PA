from typing import Dict, List

import numpy as np


class PerformanceMetrics:
    def __init__(self):
        self.scores = []
        self.episode_lengths = []
        self.episodes_per_second = 0

    def add_episode(self, score, length):
        self.scores.append(score)
        self.episode_lengths.append(length)

    def get_average_score(self, last_n=None):
        if last_n is None:
            return sum(self.scores) / len(self.scores) if self.scores else 0
        return (
            sum(self.scores[-last_n:]) / min(last_n, len(self.scores))
            if self.scores
            else 0
        )

    def get_average_length(self, last_n=None):
        if last_n is None:
            return (
                sum(self.episode_lengths) / len(self.episode_lengths)
                if self.episode_lengths
                else 0
            )
        return (
            sum(self.episode_lengths[-last_n:]) / min(last_n, len(self.episode_lengths))
            if self.episode_lengths
            else 0
        )

    def set_episodes_per_second(self, eps):
        self.episodes_per_second = eps

    def get_episodes_per_second(self):
        return self.episodes_per_second


def calculate_metrics(episode_rewards: List[float]) -> Dict[str, float]:
    rewards = np.array(episode_rewards)

    return {
        "average_reward": float(np.mean(rewards)),
        "std_dev": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "success_rate": float(np.mean(rewards > 0)),
    }
