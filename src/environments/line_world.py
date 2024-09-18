from typing import Tuple

import numpy as np

from src.environments import Environment


class LineWorld(Environment):
    def __init__(self, size: int = 7):
        self.size = size
        self.state = 0
        self.done = False
        self.score_value = 0

    def render(self) -> None:
        self.display()

    def num_states(self) -> int:
        return self.size

    def num_actions(self) -> int:
        return 2  # 0: move left, 1: move right

    def num_rewards(self) -> int:
        return 3  # -1 for moving, 0 for no move, 1 for reaching the goal

    def reward(self, i: int) -> float:
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
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
        return self.state

    def reset(self) -> int:
        self.state = 0
        self.done = False
        self.score_value = 0
        return self.state

    def display(self) -> None:
        line = ["-"] * self.size
        line[self.state] = "A"
        line[-1] = "G"
        print("".join(line))

    def is_forbidden(self, action: int) -> int:
        return 0

    def is_game_over(self) -> bool:
        return self.done

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done:
            return self.state, 0, True, {}

        if action == 0 and self.state > 0:
            self.state -= 1
            reward = -1
        elif action == 1 and self.state < self.size - 1:
            self.state += 1
            reward = -1
        else:
            reward = 0

        if self.state == self.size - 1:
            reward = 1
            self.done = True

        self.score_value += reward
        return self.state, reward, self.done, {}

    def score(self) -> float:
        return self.score_value

    @staticmethod
    def from_random_state() -> "LineWorld":
        size = np.random.randint(5, 11)
        env = LineWorld(size)
        env.state = np.random.randint(0, size)
        return env
