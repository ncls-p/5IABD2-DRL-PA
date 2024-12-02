import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from src.environments import Environment
from datetime import datetime
import os
from src.agents.reinforce import PolicyNetwork

class ValueNetwork(nn.Module):
    def __init__(self, state_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a scalar value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)  # Squeeze to produce a single value


class REINFORCEBaselineLearnedByCriticAgent:
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
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.device = device

        # Policy network (Actor)
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)

        # Value network (Critic)
        self.value_network = ValueNetwork(state_size).to(device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr_value)

    def choose_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def train_episode(self, rewards: list, log_probs: list, states: list) -> None:
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        log_probs_tensor = torch.stack(log_probs)
        states_tensor = torch.FloatTensor(states).to(self.device)

        # Compute returns and advantages
        returns = torch.zeros_like(rewards_tensor)
        future_return = 0
        for t in reversed(range(len(rewards))):
            future_return = rewards[t] + self.gamma * future_return
            returns[t] = future_return

        # Compute value estimates and advantages
        values = self.value_network(states_tensor)
        advantages = returns - values

        # Update value network (Critic)
        value_loss = advantages.pow(2).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy network (Actor)
        policy_loss = -(log_probs_tensor * advantages.detach()).sum()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

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

    def save_model(self, env_name: str) -> None:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        policy_path = os.path.join('model', 'reinforce_baseline_learned_by_critic', env_name, current_time, 'policy.pt')
        value_path = os.path.join('model', 'reinforce_baseline_learned_by_critic', env_name, current_time, 'value.pt')
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)

        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)
