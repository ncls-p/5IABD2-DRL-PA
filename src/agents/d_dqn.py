import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from typing import Any
from matplotlib import pyplot as plt
from src.environments import Environment


# Q-Network for estimating Q-values
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Double DQN Agent
class DoubleDQNAgent:
    def __init__(
        self,
        env: Environment,
        state_size: int,
        action_size: int,
        gamma=0.99,
        lr=0.001,
        target_update_freq=10,
    ):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.target_update_freq = target_update_freq

        # Policy and target networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize target network parameters to match the policy network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.env.available_actions())
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(
        self,
        num_episodes=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        scores = []
        steps_per_episode = []
        action_times = []
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.env.state_vector()
            done = False
            score = 0
            episode_steps = 0
            episode_action_times = []

            while not done:
                # Measure action execution time
                action_start = time.time()
                action = self.choose_action(state, epsilon)
                action_time = time.time() - action_start
                episode_action_times.append(action_time)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.state_vector()

                # Learn directly from the experience
                self.learn(state, action, reward, next_state, done)

                state = next_state
                score += reward
                episode_steps += 1

            # Decrease epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)
            steps_per_episode.append(episode_steps)
            action_times.append(np.mean(episode_action_times))

            # Update the target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % 100 == 0:
                avg_score = np.mean(scores[-100:])
                avg_steps = np.mean(steps_per_episode[-100:])
                avg_action_time = np.mean(action_times[-100:])
                print(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Avg Steps: {avg_steps:.2f}, "
                    f"Avg Action Time: {avg_action_time:.5f}s, "
                    f"Epsilon: {epsilon:.2f}"
                )

        return scores, steps_per_episode, action_times

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Get current Q-value estimate
        q_value = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze()

        # Double DQN target calculation
        with torch.no_grad():
            best_action = self.q_network(next_state).argmax(1)
            next_q_value = (
                self.target_network(next_state)
                .gather(1, best_action.unsqueeze(1))
                .squeeze()
            )
            target = reward + (1 - done) * self.gamma * next_q_value

        # Compute loss and optimize
        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_metrics(data, metric_name, window_size=100):
    plt.plot(data, alpha=0.6, label="Raw")
    if len(data) >= window_size:
        ma = moving_average(data, window_size)
        plt.plot(
            range(window_size - 1, len(data)), ma, label=f"{window_size}-episode MA"
        )
    plt.title(f"{metric_name} over Episodes")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
