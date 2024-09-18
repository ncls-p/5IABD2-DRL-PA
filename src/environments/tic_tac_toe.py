from typing import Tuple
import numpy as np
from src.environments import Environment


class TicTacToe(Environment):
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.score_value = 0

    def render(self) -> None:
        self.display()

    def num_states(self) -> int:
        return 3**9  # 3 possibilities for each cell, 9 cells

    def num_actions(self) -> int:
        return 9  # 9 possible moves

    def num_rewards(self) -> int:
        return 3  # -1 for loss, 0 for draw, 1 for win

    def reward(self, i: int) -> float:
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        board = self._int_to_board(s)
        next_board = self._int_to_board(s_p)
        row, col = divmod(a, 3)

        # Check if the move is invalid (cell is already occupied)
        if board[row, col] != 0:
            return 1.0 if s == s_p and r_index == 1 else 0.0

        # Make the move
        board[row, col] = self.current_player

        if np.array_equal(board, next_board):
            if self._check_win(board):
                return 1.0 if r_index == 2 else 0.0
            elif np.all(board != 0):
                return 1.0 if r_index == 1 else 0.0
            else:
                return 1.0 if r_index == 0 else 0.0
        return 0.0

    def state_id(self) -> int:
        return self._board_to_int(self.board)

    def reset(self) -> int:
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.score_value = 0
        return self.state_id()

    def display(self) -> None:
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))

    def is_forbidden(self, action: int) -> int:
        row, col = divmod(action, 3)
        return int(self.board[row, col] != 0)

    def is_game_over(self) -> bool:
        return self.done

    def available_actions(self) -> np.ndarray:
        return np.where(self.board.flatten() == 0)[0]

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        if self.done or self.is_forbidden(action):
            return self.state_id(), 0, True, {}

        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player

        if self._check_win(self.board):
            reward = 1 if self.current_player == 1 else -1
            self.done = True
        elif np.all(self.board != 0):
            reward = 0
            self.done = True
        else:
            reward = 0
            self.current_player = 3 - self.current_player  # Switch player

        self.score_value += reward
        return self.state_id(), reward, self.done, {}

    def score(self) -> float:
        return self.score_value

    @staticmethod
    def from_random_state() -> "TicTacToe":
        env = TicTacToe()
        num_moves = np.random.randint(0, 5)
        for _ in range(num_moves):
            available = env.available_actions()
            if len(available) > 0:
                action = np.random.choice(available)
                env.step(action)
            else:
                break
        return env

    def _check_win(self, board) -> bool:
        for player in [1, 2]:
            # Check rows and columns
            for i in range(3):
                if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                    return True
            # Check diagonals
            if np.all(np.diag(board) == player) or np.all(
                np.diag(np.fliplr(board)) == player
            ):
                return True
        return False

    def _int_to_board(self, s: int) -> np.ndarray:
        return np.array([(s // (3**i)) % 3 for i in range(9)]).reshape(3, 3)

    def _board_to_int(self, board: np.ndarray) -> int:
        return sum(3**i * v for i, v in enumerate(board.flatten()))
