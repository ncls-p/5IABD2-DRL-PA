import numpy as np
import random
import time
from src.environments import Environment
import pickle


class TabularQLearningAgent:
    def __init__(
        self,
        env: Environment,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.env = env
        self.state_size = self.env.num_states()
        self.action_size = self.env.num_actions()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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
        steps_per_episode = []
        action_times = []

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            score = 0
            steps = 0
            episode_action_times = []

            while not done:
                start_time = time.time()
                action = self.choose_action(state)
                episode_action_times.append(time.time() - start_time)

                next_state, reward, done, _ = self.env.step(action)
                steps += 1

                best_next_action = np.argmax(self.q_table[next_state])
                td_target = (
                    reward + self.gamma * self.q_table[next_state, best_next_action]
                )
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.learning_rate * td_error

                state = next_state
                score += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            scores.append(score)
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))

            if (episode + 1) % 1 == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Score: {np.mean(scores[-100:]):.2f}, "
                    f"Steps: {steps}, "
                    f"Avg Action Time: {np.mean(episode_action_times):.4f}s"
                )

        filepath = f"model/tab_q_learn/q_table_{self.env.env_name()}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

        return scores, steps_per_episode, action_times

    def test(self, num_episodes=100):
        """Test the agent without exploration."""
        total_score = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)
                score += reward
            total_score += score

        avg_score = total_score / num_episodes
        print(f"Average score over {num_episodes} test episodes: {avg_score}")
        return avg_score
