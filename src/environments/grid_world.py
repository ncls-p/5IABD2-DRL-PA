from typing import Tuple

import numpy as np

from src.environments import Environment


class GridWorld(Environment):
    def __init__(self, size: int = 4):
        """Initialize the GridWorld environment."""
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.lose = (size - 1, 0)
        self.wall = (1, 1)
        self.done = False
        self.score_value = 0

    def render(self) -> None:
        """Render the current state of the environment."""
        self.display()

    def num_states(self) -> int:
        """Get the number of states in the environment."""
        return self.size * self.size

    def num_actions(self) -> int:
        """Get the number of possible actions in the environment."""
        return 4  # 0: up, 1: right, 2: down, 3: left

    def num_rewards(self) -> int:
        """Get the number of possible rewards in the environment."""
        return 3

    def reward(self, i: int) -> float:
        """Get the reward value for a given reward index."""
        rewards = [-1.0, 0.0, 1.0]  # -1: move, 0: invalid move, 1: goal
        return rewards[i]

    def p(
        self, s: Tuple[int, int], a: int, s_p: Tuple[int, int], r_index: int
    ) -> float:
        """Get the transition probability for a given state, action, next state, and reward index."""
        x, y = s
        x_p, y_p = s_p

        if self.is_game_over():
            return 0.0

        if a == 0:  # up
            x_new, y_new = x, y - 1
        elif a == 1:  # right
            x_new, y_new = x + 1, y
        elif a == 2:  # down
            x_new, y_new = x, y + 1
        elif a == 3:  # left
            x_new, y_new = x - 1, y
        else:
            return 0.0

        if 0 <= x_new < self.size and 0 <= y_new < self.size:
            if (x_new, y_new) == (x_p, y_p):
                if (x_p, y_p) == self.goal:
                    return 1.0 if r_index == 2 else 0.0
                else:
                    return 1.0 if r_index == 0 else 0.0
            else:
                return 0.0
        else:
            # Invalid move, stays in the same state
            return 1.0 if (x_p, y_p) == (x, y) and r_index == 1 else 0.0

    def state_id(self) -> int:
        """Get the current state ID."""
        x, y = self.state
        return y * self.size + x

    def reset(self) -> int:
        """Reset the environment to its initial state."""
        self.state = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.done = False
        self.score_value = 0
        return self.state_id()

    def display(self) -> None:
        """Display the current state of the environment."""
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
        """Check if an action is forbidden in the current state."""
        x, y = self.state
        if action == 0 and y == 0:  # up
            return 1
        elif action == 1 and x == self.size - 1:  # right
            return 1
        elif action == 2 and y == self.size - 1:  # down
            return 1
        elif action == 3 and x == 0:  # left
            return 1
        elif action == 0 and (x, y - 1) == self.wall:  # wall -> up
            return 1
        elif action == 1 and (x + 1, y) == self.wall:  # wall -> right
            return 1
        elif action == 2 and (x, y + 1) == self.wall:  # wall -> down
            return 1
        elif action == 3 and (x - 1, y) == self.wall:  # wall -> left
            return 1
        else:
            return 0

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.done

    def available_actions(self) -> np.ndarray:
        """Get the available actions in the current state."""
        return np.array(
            [a for a in range(self.num_actions()) if not self.is_forbidden(a)]
        )

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take a step in the environment based on the given action."""
        if self.done:
            return self.state_id(), 0.0, True, {}

        if self.is_forbidden(action):
            # Invalid move
            reward = 0.0
            next_state = self.state
        else:
            x, y = self.state
            if action == 0:  # up
                y -= 1
            elif action == 1:  # right
                x += 1
            elif action == 2:  # down
                y += 1
            elif action == 3:  # left
                x -= 1
            next_state = (x, y)

            if next_state == self.lose:
                reward = -1.0
                self.done = True

            if next_state == self.goal:
                reward = 1.0
                self.done = True
            else:
                reward = 0

            self.state = next_state

        self.score_value += reward
        return self.state_id(), reward, self.done, {}

    def score(self) -> float:
        """Get the current score value."""
        return self.score_value

    @staticmethod
    def from_random_state() -> "GridWorld":
        """Create a GridWorld environment with a random size and state."""
        size = np.random.randint(3, 6)
        env = GridWorld(size)
        env.state = (np.random.randint(0, size), np.random.randint(0, size))
        env.goal = (np.random.randint(0, size), np.random.randint(0, size))
        while env.goal == env.state:
            env.goal = (np.random.randint(0, size), np.random.randint(0, size))
        return env

    def set_state(self, new_state: Tuple[int, int]) -> None:
        """Set the current state of the environment."""
        if 0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size:
            self.state = new_state
        else:
            raise ValueError(f"Invalid state: {new_state}")

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the game as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        state_vector = np.zeros(self.size * self.size, dtype=int)
        state_vector[self.state[1] * self.size + self.state[0]] = 1
        goal_vector = np.zeros(self.size * self.size, dtype=int)
        goal_vector[self.goal[1] * self.size + self.goal[0]] = 1
        return np.concatenate([state_vector, goal_vector])

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
