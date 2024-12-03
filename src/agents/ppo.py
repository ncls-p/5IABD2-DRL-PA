import logging
import time
import os
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim.adam import Adam

from src.environments import Environment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PPOPolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class PPOValueNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    def __init__(
        self,
        env: Environment,
        lr: float = 0.001,
        gamma: float = 0.99,
        lamda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 2048,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.lamda = lamda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.device = device

        state_size = env.num_states()
        action_size = env.num_actions()

        self.policy = PPOPolicyNetwork(state_size, action_size).to(device)
        self.value = PPOValueNetwork(state_size).to(device)
        self.optimizer = Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

    def choose_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        state = state.to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        log_prob = dist.log_prob(action_tensor)

        return action, log_prob

    def save_model(self, episode: int, save_dir: str = "./checkpoints"):
        """Save model weights to disk"""
        model_name = "ppo"
        env_name = self.env.__class__.__name__.lower()
        save_dir = os.path.join(save_dir, model_name, env_name)
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        save_path = os.path.join(save_dir, f"checkpoint_episode_{episode}.pt")
        torch.save(checkpoint, save_path)

        checkpoints = sorted(
            glob.glob(os.path.join(save_dir, "checkpoint_episode_*.pt"))
        )
        if len(checkpoints) > 10:
            for checkpoint_to_delete in checkpoints[:-10]:
                os.remove(checkpoint_to_delete)

    def train(
        self, num_episodes: int = 1000
    ) -> tuple[list[float], list[int], list[float]]:
        episode_rewards = []
        steps_per_episode = []
        action_times = []
        log_frequency = 1

        checkpoint_frequency = 1000

        for episode in range(num_episodes):
            state = self.env.reset()
            state_id = self.env.state_id()
            state = torch.tensor([state_id], dtype=torch.long).to(self.device)
            state = torch.nn.functional.one_hot(
                state, num_classes=self.env.num_states()
            ).float()

            episode_reward = 0
            episode_steps = 0
            episode_action_times = []

            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            done = False

            while not done:
                start_time = time.time()
                action, log_prob = self.choose_action(state)
                action_time = time.time() - start_time
                episode_action_times.append(action_time)

                next_state, reward, done, _ = self.env.step(action)
                next_state_id = self.env.state_id()
                next_state = torch.tensor([next_state_id], dtype=torch.long).to(
                    self.device
                )
                next_state = torch.nn.functional.one_hot(
                    next_state, num_classes=self.env.num_states()
                ).float()

                states.append(state)
                actions.append(torch.tensor(action).to(self.device))
                rewards.append(
                    torch.tensor(reward, dtype=torch.float32).to(self.device)
                )
                log_probs.append(log_prob.detach())
                dones.append(torch.tensor(done, dtype=torch.float32).to(self.device))

                episode_reward += reward
                episode_steps += 1
                state = next_state

            with torch.no_grad():
                batch_states = torch.stack(states).to(self.device)
                batch_values = self.value(batch_states).view(-1)
                next_value = self.value(state).item()

            batch_actions = torch.stack(actions).view(-1)
            batch_rewards = torch.stack(rewards)
            batch_dones = torch.stack(dones)
            batch_log_probs = torch.stack(log_probs)

            advantages = []
            gae = 0
            for i in reversed(range(len(batch_rewards))):
                delta = (
                    batch_rewards[i]
                    + self.gamma * next_value * (1 - batch_dones[i])
                    - batch_values[i]
                )
                gae = delta + self.gamma * self.lamda * (1 - batch_dones[i]) * gae
                advantages.insert(0, gae)
                next_value = batch_values[i]

            advantages = torch.stack(advantages).to(self.device)
            returns = advantages + batch_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(self.epochs):
                indices = np.random.permutation(len(batch_states))

                for start_idx in range(0, len(batch_states), self.batch_size):
                    end_idx = start_idx + self.batch_size
                    batch_indices = indices[start_idx:end_idx]

                    mini_batch_states = batch_states[batch_indices]
                    mini_batch_actions = batch_actions[batch_indices]
                    mini_batch_log_probs = batch_log_probs[batch_indices]
                    mini_batch_advantages = advantages[batch_indices]
                    mini_batch_returns = returns[batch_indices]

                    new_probs = self.policy(mini_batch_states)
                    dist = Categorical(new_probs)
                    new_log_probs = dist.log_prob(mini_batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - mini_batch_log_probs)
                    surrogate1 = ratio * mini_batch_advantages
                    surrogate2 = (
                        torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * mini_batch_advantages
                    )

                    policy_loss = -torch.min(surrogate1, surrogate2).mean()
                    value_loss = nn.MSELoss()(
                        self.value(mini_batch_states).view(-1), mini_batch_returns
                    )
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            episode_rewards.append(episode_reward)
            steps_per_episode.append(episode_steps)
            action_times.append(np.mean(episode_action_times))

            if (episode + 1) % log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-log_frequency:])
                avg_steps = np.mean(steps_per_episode[-log_frequency:])
                avg_action_time = np.mean(action_times[-log_frequency:])
                logger.info(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Steps: {avg_steps:.2f} | "
                    f"Avg Action Time: {avg_action_time:.4f}s"
                )

            if (episode + 1) % checkpoint_frequency == 0:
                self.save_model(episode + 1)
                logger.info(f"Saved model checkpoint at episode {episode + 1}")

        return episode_rewards, steps_per_episode, action_times
