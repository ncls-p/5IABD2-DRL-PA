import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.distributions import Categorical
from src.environments import Environment
from src.agents.reinforce import REINFORCEAgent

class ValueNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class REINFORCEwithBaselineAgent(REINFORCEAgent):
    def __init__(
        self,
        env: Environment,
        state_size: int,
        action_size: int,
        lr_policy: float = 0.001,
        lr_value: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu",
        env_name: str = "DefaultEnv"
    ):
        super().__init__(env, state_size, action_size, lr_policy, gamma, device, env_name)
        self.value_network = ValueNetwork(state_size).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr_value)

    def train_episode(self, rewards: list, log_probs: list, states: list) -> None:
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        log_probs_tensor = torch.stack(log_probs)
        states_tensor = torch.FloatTensor(states).to(self.device)

        returns = torch.zeros_like(rewards_tensor)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return

        values = self.value_network(states_tensor)
        advantages = returns - values.detach()

        value_loss = nn.MSELoss()(values, returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        policy_loss = -(log_probs_tensor * advantages).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train(self, num_episodes: int = 1000) -> list[float]:
        episode_rewards = []

        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []

            state = self.env.reset()
            state = self.env.state_vector()
            done = False
            episode_reward = 0

            while not done:
                action, log_prob = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.state_vector()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)

                state = next_state
                episode_reward += reward

            episode_rewards.append(episode_reward)

            self.train_episode(rewards, log_probs, states)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

        return episode_rewards
