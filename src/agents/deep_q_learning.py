import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from src.environments import Environment
import time

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

# Replay Buffer to store past experiences
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, env: Environment, state_size: int, action_size: int, batch_size=16, gamma=0.95, lr=0.0005, buffer_size=1000):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.buffer_size = buffer_size

        # CUDA: Check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks for policy (Q-function) and target Q-function
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copy weights from the Q-network to the target network
        self.update_target_network()

        # Replay buffer for experience replay
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.env.available_actions())
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(
        self,
        num_episodes,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        target_update_freq=10,
    ):
        scores = []
        epsilon = epsilon_start
        action_times = []
        steps_per_episode = []

        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.env.state_vector()  # Obtain vectorized state representation
            done = False
            score = 0
            episode_action_times = []
            episode_steps = 0

            while not done:
                action_start = time.time()
                action = self.choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.state_vector()
                action_time = time.time() - action_start
                episode_action_times.append(action_time)

                # Store experience in replay buffer
                self.memory.add((state, action, reward, next_state, done))
                state = next_state
                score += reward

                # Sample a random batch from the replay buffer and update the Q-network
                if len(self.memory) >= self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

                episode_steps += 1

            # Decrease epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)
            steps_per_episode.append(episode_steps)
            action_times.append(np.mean(episode_action_times))

            # Update the target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % 1 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}")

        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"model/deep_q_learn/deep_q_learn_{self.env.env_name()}.pth",
        )

        return scores, steps_per_episode, action_times

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert list of numpy arrays to numpy arrays
        states = np.array(states)  # Convert to numpy array first
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

