from typing import Tuple

import numpy as np

from src.environments import Environment


class GridWorld(Environment):
    def __init__(self, size: int = 4):
        """
        Initialize the GridWorld environment.

        Args:
            size (int): The size of the grid world.
        """
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.done = False
        self.score_value = 0

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
        return self.size * self.size

    def num_actions(self) -> int:
        """
        Get the number of possible actions in the environment.

        Returns:
            int: The number of actions (0: up, 1: right, 2: down, 3: left).
        """
        return 4

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

    def p(
        self, s: Tuple[int, int], a: int, s_p: Tuple[int, int], r_index: int
    ) -> float:
        """
        Get the transition probability for a given state, action, next state, and reward index.

        Args:
            s (Tuple[int, int]): The current state.
            a (int): The action taken.
            s_p (Tuple[int, int]): The next state.
            r_index (int): The reward index.

        Returns:
            float: The transition probability.
        """
        x, y = s
        x_p, y_p = s_p

        if (x, y) == self.goal and (x_p, y_p) == self.goal:
            return 1.0 if r_index == 1 else 0.0

        if a == 0:  # up
            valid = y > 0 and (x_p, y_p) == (x, y - 1)
        elif a == 1:  # right
            valid = x < self.size - 1 and (x_p, y_p) == (x + 1, y)
        elif a == 2:  # down
            valid = y < self.size - 1 and (x_p, y_p) == (x, y + 1)
        elif a == 3:  # left
            valid = x > 0 and (x_p, y_p) == (x - 1, y)
        else:
            return 0.0

        if valid:
            if (x_p, y_p) == self.goal:
                return 1.0 if r_index == 2 else 0.0
            else:
                return 1.0 if r_index == 0 else 0.0
        elif (x_p, y_p) == (x, y):
            return 1.0 if r_index == 1 else 0.0
        else:
            return 0.0

    def state_id(self) -> int:
        """
        Get the current state ID.

        Returns:
            int: The current state ID.
        """
        x, y = self.state
        return y * self.size + x

    def reset(self) -> int:
        """
        Reset the environment to its initial state.

        Returns:
            int: The initial state ID.
        """
        self.state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        self.goal = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        while self.goal == self.state:
            self.goal = (
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            )
        self.done = False
        self.score_value = 0
        return self.state_id()

    def display(self) -> None:
        """
        Display the current state of the environment.
        """
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if (x, y) == self.state:
                    row.append("A")
                elif (x, y) == self.goal:
                    row.append("G")
                else:
                    row.append(".")
            print(" ".join(row))

    def is_forbidden(self, action: int) -> int:
        """
        Check if an action is forbidden in the current state.

        Args:
            action (int): The action to check.

        Returns:
            int: 0 if the action is allowed, 1 if the action is forbidden.
        """
        x, y = self.state
        if action == 0 and y == 0:  # Trying to move up from the top edge
            return 1
        elif (
            action == 1 and x == self.size - 1
        ):  # Trying to move right from the right edge
            return 1
        elif (
            action == 2 and y == self.size - 1
        ):  # Trying to move down from the bottom edge
            return 1
        elif action == 3 and x == 0:  # Trying to move left from the left edge
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
        return np.array(
            [
                action
                for action in range(self.num_actions())
                if not self.is_forbidden(action)
            ]
        )

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[int, float, bool, dict]: The next state ID, reward, done flag, and additional info.
        """
        if self.done:
            return self.state_id(), 0, True, {}

        x, y = self.state
        if action == 0 and y > 0:  # up
            y -= 1
        elif action == 1 and x < self.size - 1:  # right
            x += 1
        elif action == 2 and y < self.size - 1:  # down
            y += 1
        elif action == 3 and x > 0:  # left
            x -= 1

        self.state = (x, y)
        reward = self.calculate_reward()

        if self.state == self.goal:
            reward = 100  # Large positive reward for reaching the goal
            self.done = True

        self.score_value += reward
        return self.state_id(), reward, self.done, {}

    def calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state and goal position.

        Returns:
            float: The calculated reward.
        """
        distance_to_goal = abs(self.state[0] - self.goal[0]) + abs(
            self.state[1] - self.goal[1]
        )
        return -distance_to_goal  # Negative reward based on distance to the goal

    def score(self) -> float:
        """
        Get the current score value.

        Returns:
            float: The current score value.
        """
        return self.score_value

    @staticmethod
    def from_random_state() -> "GridWorld":
        """
        Create a GridWorld environment with a random size and state.

        Returns:
            GridWorld: A new GridWorld environment with a random size and state.
        """
        size = np.random.randint(3, 6)
        env = GridWorld(size)
        env.state = (np.random.randint(0, size), np.random.randint(0, size))
        return env

    def set_state(self, new_state: Tuple[int, int]) -> None:
        """
        Set the current state of the environment.

        Args:
            new_state (Tuple[int, int]): The new state to set.

        Raises:
            ValueError: If the new state is invalid.
        """
        if 0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size:
            self.state = new_state
        else:
            raise ValueError(f"Invalid state: {new_state}")

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the environment as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        return np.array(self.state)

    def action_vector(self, action: int) -> np.ndarray:
        """
        Get the vector encoding of the given action.

        Args:
            action (int): The action to encode.

        Returns:
            np.ndarray: The vector encoding of the action.
        """
        action_vector = np.zeros(4, dtype=int)
        action_vector[action] = 1
        return action_vector