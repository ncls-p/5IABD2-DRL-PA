from typing import Tuple

import numpy as np

from src.environments import Environment


class LineWorld(Environment):
    def __init__(self, size: int = 7):
        """
        Initialize the LineWorld environment.

        Args:
            size (int): The size of the line world.
        """
        self.size = size
        self.state = 0
        self.done = False
        self.score_value = 0
        self.target_position = size - 1

    def render(self) -> None:
        """
        Render the current state of the environment.
        """
        self.display()

    def num_states(self) -> int:
        """
        Get the number of states in the environment.

        Returns:
            int: The number of states.
        """
        return self.size

    def num_actions(self) -> int:
        """
        Get the number of possible actions in the environment.

        Returns:
            int: The number of actions (0: move left, 1: move right).
        """
        return 2

    def num_rewards(self) -> int:
        """
        Get the number of possible rewards in the environment.

        Returns:
            int: The number of rewards (-1 for moving, 0 for no move, 1 for reaching the goal).
        """
        return 3

    def reward(self, i: int) -> float:
        """
        Get the reward value for a given reward index.

        Args:
            i (int): The reward index.

        Returns:
            float: The reward value.
        """
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """
        Get the transition probability for a given state, action, next state, and reward index.

        Args:
            s (int): The current state.
            a (int): The action taken.
            s_p (int): The next state.
            r_index (int): The reward index.

        Returns:
            float: The transition probability.
        """
        if s == 0 and a == 0:  # Left edge, trying to move left
            return 1.0 if s_p == s and r_index == 1 else 0.0
        elif s == self.size - 1 and a == 1:  # Right edge, trying to move right
            return 1.0 if s_p == s and r_index == 1 else 0.0
        elif a == 0 and s_p == s - 1:  # Moving left
            return 1.0 if r_index == 0 else 0.0
        elif a == 1 and s_p == s + 1:  # Moving right
            if s_p == self.size - 1:  # Reaching the goal
                return 1.0 if r_index == 2 else 0.0
            else:
                return 1.0 if r_index == 0 else 0.0
        else:
            return 0.0

    def state_id(self) -> int:
        """
        Get the current state ID.

        Returns:
            int: The current state ID.
        """
        return self.state

    def reset(self) -> int:
        """
        Reset the environment to its initial state.

        Returns:
            int: The initial state.
        """
        self.state = np.random.randint(0, self.size)
        self.target_position = np.random.randint(0, self.size)
        while self.target_position == self.state:
            self.target_position = np.random.randint(0, self.size)
        self.done = False
        self.score_value = 0
        return self.state

    def display(self) -> None:
        """
        Display the current state of the environment.
        """
        line = ["-"] * self.size
        line[self.state] = "A"
        line[self.target_position] = "G"
        print("".join(line))

    def is_forbidden(self, action: int) -> int:
        """
        Check if an action is forbidden in the current state.

        Args:
            action (int): The action to check.

        Returns:
            int: 0 if the action is allowed, 1 if the action is forbidden.
        """
        if action == 0 and self.state == 0:  # Trying to move left from the left edge
            return 1
        elif (
            action == 1 and self.state == self.size - 1
        ):  # Trying to move right from the right edge
            return 1
        return 0

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.done

    def available_actions(self) -> np.ndarray:
        """
        Get the available actions in the current state.

        Returns:
            np.ndarray: An array of available actions.
        """
        return np.array([0, 1])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[int, float, bool, dict]: The next state, reward, done flag, and additional info.
        """
        if self.done:
            return self.state, 0, True, {}

        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < self.size - 1:
            self.state += 1

        reward = self.calculate_reward()

        if self.state == self.target_position:
            reward = 100  # Large positive reward for reaching the target
            self.done = True

        self.score_value += reward
        return self.state, reward, self.done, {}

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state and target position.

        Returns:
            float: The calculated reward.
        """
        distance_to_target = abs(self.state - self.target_position)
        return -distance_to_target  # Negative reward based on distance to the target

    def score(self) -> float:
        """
        Get the current score value.

        Returns:
            float: The current score value.
        """
        return self.score_value

    @staticmethod
    def from_random_state() -> "LineWorld":
        """
        Create a LineWorld environment with a random size and state.

        Returns:
            LineWorld: A new LineWorld environment with a random size and state.
        """
        size = np.random.randint(5, 11)
        env = LineWorld(size)
        env.state = np.random.randint(0, size)
        return env

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the environment as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        return np.array([self.state])

    def action_vector(self, action: int) -> np.ndarray:
        """
        Get the vector encoding of the given action.

        Args:
            action (int): The action to encode.

        Returns:
            np.ndarray: The vector encoding of the action.
        """
        action_vector = np.zeros(2, dtype=int)
        action_vector[action] = 1
        return action_vector