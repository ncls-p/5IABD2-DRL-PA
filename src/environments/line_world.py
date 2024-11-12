from typing import Tuple

import numpy as np

from src.environments import Environment


class LineWorld(Environment):
    def __init__(self, size: int = 7):
        """Initialize the LineWorld environment."""
        self.size = size
        self.state = 0
        self.done = False
        self.score_value = 0
        self.target_position = size - 1

    def render(self) -> None:
        """Render the current state of the environment."""
        self.display()

    def num_states(self) -> int:
        """Get the number of states in the environment."""
        return self.size

    def num_actions(self) -> int:
        """Get the number of possible actions in the environment."""
        return 2  # 0: left, 1: right

    def num_rewards(self) -> int:
        """Get the number of possible rewards in the environment."""
        return 3

    def reward(self, i: int) -> float:
        """Get the reward value for a given reward index."""
        rewards = [-1.0, 0.0, 1.0]  # -1: move, 0: invalid move, 1: goal
        return rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """Get the transition probability for a given state, action, next state, and reward index."""
        if self.is_game_over():
            return 0.0

        if a == 0:  # left
            s_new = s - 1
        elif a == 1:  # right
            s_new = s + 1
        else:
            return 0.0

        if 0 <= s_new < self.size:
            if s_new == s_p:
                if s_p == self.target_position:
                    return 1.0 if r_index == 2 else 0.0
                else:
                    return 1.0 if r_index == 0 else 0.0
            else:
                return 0.0
        else:
            # Invalid move
            return 1.0 if s_p == s and r_index == 1 else 0.0

    def state_id(self) -> int:
        """Get the current state ID."""
        return self.state

    def reset(self) -> int:
        """Reset the environment to its initial state."""
        self.state = 0
        self.target_position = self.size - 1
        self.done = False
        self.score_value = 0
        return self.state

    def display(self) -> None:
        """Display the current state of the environment."""
        line = ["-"] * self.size
        line[self.state] = "A"
        line[self.target_position] = "G"
        print("".join(line))

    def is_forbidden(self, action: int) -> int:
        """Check if an action is forbidden in the current state."""
        if action == 0 and self.state == 0:  # left
            return 1
        elif action == 1 and self.state == self.size - 1:  # right
            return 1
        else:
            return 0

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.done

    def available_actions(self) -> np.ndarray:
        """Get the available actions in the current state."""
        return np.array([a for a in range(2) if not self.is_forbidden(a)])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take a step in the environment based on the given action."""
        if self.done:
            return self.state, 0.0, True, {}

        if self.is_forbidden(action):
            # Invalid move
            reward = 0.0
        else:
            if action == 0:  # left
                self.state -= 1
            elif action == 1:  # right
                self.state += 1

            if self.state == self.target_position:
                reward = 1.0
                self.done = True
            elif self.state == 0:
                reward = -1
                self.done = True
            else:
                reward = 0

        self.score_value += reward
        return self.state_id(), reward, self.done, {}

    def score(self) -> float:
        """Get the current score value."""
        return self.score_value

    @staticmethod
    def from_random_state() -> "LineWorld":
        """Create a LineWorld environment with a random size and state."""
        size = np.random.randint(5, 11)
        env = LineWorld(size)
        env.state = np.random.randint(0, size)
        env.target_position = np.random.randint(0, size)
        while env.target_position == env.state:
            env.target_position = np.random.randint(0, size)
        return env

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the game as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        state_vector = np.zeros(self.size, dtype=int)
        state_vector[self.state] = 1
        state_vector = np.concatenate([state_vector, [self.target_position]])
        return state_vector

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
