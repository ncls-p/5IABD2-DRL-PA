import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Any, List, Tuple
from src.environments import Environment
import time

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_list_to_tensor(data_list, dtype=torch.float32, device=device):
        # Convert list to numpy array and then to a tensor
        return torch.tensor(np.array(data_list), dtype=dtype, device=device)

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

# Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64, alpha=0.6):
        self.buffer = []
        self.priorities = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.position = 0
        self.alpha = alpha

    def add(self, experience, td_error):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        # Set priority as max or based on TD-error
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.priorities) < self.buffer_size:
            self.priorities.append(priority)
        else:
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.buffer_size

    def sample(self, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** -beta
        weights /= weights.max()  # Normalize for stability

        return experiences, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)

# Double DQN Agent with Prioritized Experience Replay
class DoubleDQNAgent:
    def __init__(self, env: Environment, state_size: int, action_size: int, gamma=0.99, lr=0.001, buffer_size=10000, batch_size=64, alpha=0.6, beta=0.4):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta

        # Policy and target networks
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize target network parameters to match the policy network
        self.update_target_network()

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, alpha=alpha)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.env.available_actions())
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Move state to GPU
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def train(self, num_episodes=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=10):
        scores = []
        steps_per_episode = []
        action_times = []
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.env.state_vector()
            done = False
            score = 0
            steps = 0
            episode_action_times = []

            while not done:
                start_time = time.time()
                action = self.choose_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.env.state_vector()
                steps += 1

                # TD-error as initial priority
                with torch.no_grad():
                    td_error = reward + self.gamma * self.q_network(torch.FloatTensor(next_state).unsqueeze(0).to(device)).max(1)[0].item() - \
                               self.q_network(torch.FloatTensor(state).unsqueeze(0).to(device))[0, action].item()

                self.memory.add((state, action, reward, next_state, done), td_error)

                state = next_state
                score += reward

                # Learn if enough samples in memory
                if len(self.memory) >= self.batch_size:
                    experiences, indices, weights = self.memory.sample(self.beta)
                    self.learn(experiences, indices, weights)

                episode_action_times.append(time.time() - start_time)

            # Decrease epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))

            # Update the target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}")

        return scores, steps_per_episode, action_times

    def learn(self, experiences, indices, weights):
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert lists to NumPy arrays first, then to tensors
        states = convert_list_to_tensor(states, dtype=torch.float32)
        actions = convert_list_to_tensor(actions, dtype=torch.long).unsqueeze(1)  # Make sure actions is (batch_size, 1)
        rewards = convert_list_to_tensor(rewards, dtype=torch.float32)
        next_states = convert_list_to_tensor(next_states, dtype=torch.float32)
        dones = convert_list_to_tensor(dones, dtype=torch.float32)
        weights = weights.unsqueeze(1).to(device)

        # Compute Q-values and Double DQN targets
        q_values = self.q_network(states).gather(1, actions)  # actions now has shape (batch_size, 1)

        with torch.no_grad():
            best_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, best_actions).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute and apply loss with importance-sampling weights
        td_errors = (q_values.squeeze() - targets).detach().cpu().numpy()
        loss = (weights * (q_values - targets.unsqueeze(1)) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in the buffer
        self.memory.update_priorities(indices, td_errors)
