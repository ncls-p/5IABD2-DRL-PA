import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
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

        # Neural networks for policy (Q-function) and target Q-function
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
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
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, num_episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999, target_update_freq=10):
        scores = []
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.env.state_vector()  # Obtain vectorized state representation
            done = False
            score = 0

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.state_vector()
                #print(len(state))

                # Store experience in replay buffer
                self.memory.add((state, action, reward, next_state, done))
                state = next_state
                score += reward

                # Sample a random batch from the replay buffer and update the Q-network
                if len(self.memory) >= self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

            # Decrease epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)

            # Update the target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.5f}"
                )

        return scores

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        #print(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze()

        # Compute target Q-values
        next_q_values = self.target_network(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and backpropagation
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
