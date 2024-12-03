import torch
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Union
from src.environments import Environment
from copy import deepcopy

class RandomRolloutAgent:
    def __init__(
        self,
        env: Environment,
        num_rollouts: int = 100,
        gamma: float = 0.99,
        env_name: str = "DefaultEnv",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.num_rollouts = num_rollouts
        self.gamma = gamma
        self.env_name = env_name
        self.device = device
        self.best_episode_reward = float('-inf')
        self.best_sequence = {'states': [], 'actions': [], 'rewards': []}

    def copy_environment(self, env: Environment) -> Environment:
        return deepcopy(env)

    def simulate(self, env: Environment, state: torch.Tensor, max_steps: int = 100) -> float:
        """Effectue une simulation aléatoire à partir d'un état"""
        sim_env = self.copy_environment(env)
        done = False
        total_reward = 0
        discount = 1.0
        steps = 0

        while not done and steps < max_steps:
            available_actions = sim_env.available_actions()
            if len(available_actions) == 0:
                break

            action = np.random.choice(available_actions)
            _, reward, done, _ = sim_env.step(action)

            total_reward += discount * reward
            discount *= self.gamma
            steps += 1

        return total_reward

    def choose_action(self, state: torch.Tensor) -> int:
        """Choisit la meilleure action basée sur plusieurs simulations"""
        available_actions = self.env.available_actions()
        if len(available_actions) == 1:
            return available_actions[0]

        action_values = []

        for action in available_actions:
            sim_env = self.copy_environment(self.env)
            sim_env.step(action)

            action_value = 0
            for _ in range(self.num_rollouts):
                value = self.simulate(sim_env, state)
                action_value += value
            action_values.append(action_value / self.num_rollouts)

        return available_actions[np.argmax(action_values)]

    def train(
        self, num_episodes: int = 10000, max_steps: int = 1000
    ) -> Tuple[List[float], List[int], List[float]]:
        scores, steps_per_episode, action_times = [], [], []
        writer = self._create_tensorboard_writer()

        for episode in range(num_episodes):
            state = self.env.reset()
            state = torch.tensor(self.env.state_vector(), device=self.device, dtype=torch.float32)
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_action_times = []

            states = [state]
            actions = []
            rewards = []

            while not done and episode_steps < max_steps:
                start_time = time.time()
                action = self.choose_action(state)
                action_time = time.time() - start_time
                episode_action_times.append(action_time)

                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(self.env.state_vector(), device=self.device, dtype=torch.float32)

                states.append(next_state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                self.best_sequence = {
                    'states': torch.stack(states),
                    'actions': torch.tensor(actions, device=self.device),
                    'rewards': torch.tensor(rewards, device=self.device)
                }

            self._log_metrics(writer, episode_reward, episode_steps, episode_action_times, episode)
            self._save_checkpoint(episode, episode_reward)

            scores.append(episode_reward)
            steps_per_episode.append(episode_steps)
            action_times.append(np.mean(episode_action_times))

            if (episode + 1) % 1 == 0:
                avg_score = (
                    np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                )
                avg_steps = (
                    np.mean(steps_per_episode[-100:])
                    if len(steps_per_episode) >= 100
                    else np.mean(steps_per_episode)
                )
                avg_time = (
                    np.mean(action_times[-100:])
                    if len(action_times) >= 100
                    else np.mean(action_times)
                )
                print(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Avg Steps: {avg_steps:.2f}, "
                    f"Avg Time/Step: {avg_time*1000:.2f}ms, "
                    f"Best Score: {self.best_episode_reward:.2f}"
                )

        writer.close()
        return scores, steps_per_episode, action_times

    def _create_tensorboard_writer(self) -> SummaryWriter:
        log_dir = os.path.join('runs', 'random_rollout', self.env_name, f'{datetime.now():%Y%m%d-%H%M%S}')
        return SummaryWriter(log_dir)

    def _log_metrics(self, writer: SummaryWriter, episode_reward: float, episode_steps: int, episode_action_times: List[float], episode: int) -> None:
        writer.add_scalar('Metrics/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Metrics/Steps_per_Episode', episode_steps, episode)
        writer.add_scalar('Metrics/Average_Action_Time', np.mean(episode_action_times), episode)

    def _save_checkpoint(self, episode: int, episode_reward: float) -> None:
        if (episode + 1) % 10000 == 0:
            checkpoint_path = os.path.join('model', 'random_rollout',
                                         self.env_name, f'checkpoint_{episode+1}.pt')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'episode': episode + 1,
                'sequence': self.best_sequence,
                'total_reward': self.best_episode_reward,
                'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
            }
            torch.save(checkpoint, checkpoint_path)
