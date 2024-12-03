import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Any, List, Tuple
from src.environments import Environment
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_list_to_tensor(data_list, dtype=torch.float32, device=device):
    return torch.tensor(np.array(data_list), dtype=dtype, device=device)


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

        total = len(self.buffer)
        weights = (total * probs[indices]) ** -beta
        weights /= weights.max()

        return experiences, indices, torch.FloatTensor(weights).to(device)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    def __init__(self, env: Environment, state_size: int, action_size: int, gamma=0.99, lr=0.001, buffer_size=10000, batch_size=64, alpha=0.6, beta=0.4):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta

        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()

        self.memory = PrioritizedReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size, alpha=alpha
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, state, epsilon=0.1):
        available_actions = self.env.available_actions()

        if random.random() < epsilon:
            return np.random.choice(available_actions)

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)

            action_idx = q_values.argmax(1).item() - 1
            if action_idx not in available_actions:
                return np.random.choice(available_actions)
            return action_idx

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

                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    next_state_tensor = (
                        torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    )
                    action_tensor = torch.tensor(action + 1).to(device)
                    td_error = (
                        reward
                        + self.gamma
                        * self.q_network(next_state_tensor).max(1)[0].item()
                        - self.q_network(state_tensor)[0, action_tensor].item()
                    )

                self.memory.add((state, action, reward, next_state, done), td_error)

                state = next_state
                score += reward

                if len(self.memory) >= self.batch_size:
                    experiences, indices, weights = self.memory.sample(self.beta)
                    self.learn(experiences, indices, weights)

                episode_action_times.append(time.time() - start_time)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            scores.append(score)
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))

            if episode % target_update_freq == 0:
                self.update_target_network()

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
                    f"Epsilon: {epsilon:.3f}"
                )

        return scores, steps_per_episode, action_times

    def learn(self, experiences, indices, weights):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = convert_list_to_tensor(states, dtype=torch.float32)
        actions = convert_list_to_tensor(
            [a + 1 for a in actions], dtype=torch.long
        ).unsqueeze(1)
        rewards = convert_list_to_tensor(rewards, dtype=torch.float32)
        next_states = convert_list_to_tensor(next_states, dtype=torch.float32)
        dones = convert_list_to_tensor(dones, dtype=torch.float32)
        weights = weights.unsqueeze(1).to(device)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            best_actions = (self.q_network(next_states).argmax(1)).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, best_actions).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        td_errors = (q_values.squeeze() - targets).detach().cpu().numpy()
        loss = (weights * (q_values - targets.unsqueeze(1)) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
