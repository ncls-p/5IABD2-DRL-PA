import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from src.environments import Environment
from datetime import datetime
import os

class PolicyNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class REINFORCEAgent:
    def __init__(
        self,
        env: Environment,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        device: str = "cpu",
        env_name: str = "DefaultEnv"
    ):
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.device = device
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train_episode(self, rewards: list, log_probs: list) -> None:
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = torch.stack([-log_prob * G for log_prob, G in zip(log_probs, returns)])
        policy_loss = policy_loss.sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def choose_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

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

            self.train_episode(rewards, log_probs)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

            # Save model checkpoint every 10000 episodes
            if (episode + 1) % 10000 == 0:
                checkpoint_path = os.path.join('model', 'reinforce',
                                             self.env_name,
                                             f'checkpoint_{episode+1}.pt')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

                # Save checkpoint with more information
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'rewards': episode_rewards,
                    'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S')
                }, checkpoint_path)

                print(f"Saved checkpoint at episode {episode + 1}")

        # Save final model
        final_path = os.path.join('model', 'reinforce',
                                 self.env_name,
                                 f'final_model.pt')
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(self.policy.state_dict(), final_path)

        return episode_rewards

    def save_model(self, env_name: str) -> None:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        path = os.path.join('model', 'reinforce', env_name, current_time, 'policy.pt')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.policy.state_dict(), path)
