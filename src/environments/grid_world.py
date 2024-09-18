from typing import Tuple
import numpy as np
from src.environments import Environment


class GridWorld(Environment):
    def __init__(self, size: int = 4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)
        self.done = False
        self.score_value = 0

    def render(self) -> None:
        self.display()

    def num_states(self) -> int:
        return self.size * self.size

    def num_actions(self) -> int:
        return 4  # 0: up, 1: right, 2: down, 3: left

    def num_rewards(self) -> int:
        return 3  # -1 for moving, 0 for no move, 1 for reaching the goal

    def reward(self, i: int) -> float:
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]

    def p(
        self, s: Tuple[int, int], a: int, s_p: Tuple[int, int], r_index: int
    ) -> float:
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
        x, y = self.state
        return y * self.size + x

    def reset(self) -> int:
        self.state = (0, 0)
        self.done = False
        self.score_value = 0
        return self.state_id()

    def display(self) -> None:
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
        return 0

    def is_game_over(self) -> bool:
        return self.done

    def available_actions(self) -> np.ndarray:
        return np.array(
            [
                action
                for action in range(self.num_actions())
                if not self.is_forbidden(action)
            ]
        )

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
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
        reward = -1

        if self.state == self.goal:
            reward = 1
            self.done = True

        self.score_value += reward
        return self.state_id(), reward, self.done, {}

    def score(self) -> float:
        return self.score_value

    @staticmethod
    def from_random_state() -> "GridWorld":
        size = np.random.randint(3, 6)
        env = GridWorld(size)
        env.state = (np.random.randint(0, size), np.random.randint(0, size))
        return env

    def set_state(self, new_state: Tuple[int, int]) -> None:
        if 0 <= new_state[0] < self.size and 0 <= new_state[1] < self.size:
            self.state = new_state
        else:
            raise ValueError(f"Invalid state: {new_state}")
