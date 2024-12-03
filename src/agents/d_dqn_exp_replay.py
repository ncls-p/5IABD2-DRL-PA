import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.environments import Environment
import time


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class DoubleDQNAgent:
    def __init__(
        self,
        env: Environment,
        state_size: int,
        action_size: int,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        buffer_size=10000,
    ):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.buffer_size = buffer_size

        # Neural networks for policy (Q-function) and target Q-function
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copy weights from the Q-network to the target network initially
        self.update_target_network()

        # Replay buffer for experience replay
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

    def update_target_network(self):
        """Update target network to have the same weights as the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        available_actions = self.env.available_actions()

        if random.random() < epsilon:
            return np.random.choice(available_actions)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
            # Subtract 1 from action index to convert back to environment's action space
            action_idx = q_values.argmax(1).item() - 1
            # If the chosen action is not available, randomly choose from available actions
            if action_idx not in available_actions:
                return np.random.choice(available_actions)
            return action_idx

    def train(
        self,
        num_episodes=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10,
    ):
        scores = []
        epsilons = []
        steps_per_episode = []
        action_times = []
        epsilon = epsilon_start

        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.env.state_vector()  # Obtain vectorized state representation
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

                # Store experience in replay buffer
                self.memory.add((state, action, reward, next_state, done))
                state = next_state
                score += reward

                # Sample a random batch from the replay buffer and update the Q-network
                if len(self.memory) >= self.batch_size:
                    experiences = self.memory.sample()
                    self.learn(experiences)

                episode_action_times.append(time.time() - start_time)

            # Decrease epsilon after each episode
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)
            epsilons.append(epsilon)
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))


            # Update the target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()

            if (episode + 1) % 1 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}"
                )

        return scores, steps_per_episode, action_times

    def learn(self, experiences):
        """Update the Q-network with Double DQN targets."""
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to numpy arrays first, then to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        # Add 1 to actions to handle negative indices (-1 becomes 0, 0 becomes 1, etc.)
        actions = torch.LongTensor(np.array(actions) + 1).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        # Get current Q values
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions).squeeze()

        # Get next Q values using Double DQN
        with torch.no_grad():
            # Use Q-network to select actions
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Use target network to evaluate actions
            next_q_values = (
                self.target_network(next_states).gather(1, next_actions).squeeze()
            )
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
