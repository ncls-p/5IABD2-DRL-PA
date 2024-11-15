import numpy as np
import random
from src.environments import Environment

class TabularQLearningAgent:
    def __init__(self, env: Environment, learning_rate=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.state_size = self.env.num_states()
        self.action_size = self.env.num_actions()
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon_start  # initial exploration rate
        self.epsilon_min = epsilon_min  # minimum exploration rate
        self.epsilon_decay = epsilon_decay  # exploration decay rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.env.available_actions())
        else:
            return np.argmax(self.q_table[state])

    def train(self, num_episodes=1000):
        """Train the agent over multiple episodes."""
        scores = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Q-learning update rule
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.learning_rate * td_error

                state = next_state
                score += reward

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            scores.append(score)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}")

        return scores

    def test(self, num_episodes=100):
        """Test the agent without exploration."""
        total_score = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = np.argmax(self.q_table[state])  # Exploit learned policy
                state, reward, done, _ = self.env.step(action)
                score += reward
            total_score += score

        avg_score = total_score / num_episodes
        print(f"Average score over {num_episodes} test episodes: {avg_score}")
        return avg_score
