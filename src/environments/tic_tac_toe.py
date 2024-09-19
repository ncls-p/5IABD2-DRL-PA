from typing import Tuple

import numpy as np

from src.environments import Environment


class TicTacToe(Environment):
    def __init__(self):
        """
        Initialize the TicTacToe environment.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
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
            int: The number of states (3 possibilities for each cell, 9 cells).
        """
        return 3**9

    def num_actions(self) -> int:
        """
        Get the number of possible actions in the environment.

        Returns:
            int: The number of actions (9 possible moves).
        """
        return 9

    def num_rewards(self) -> int:
        """
        Get the number of possible rewards in the environment.

        Returns:
            int: The number of rewards (-1 for loss, 0 for draw, 1 for win).
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
        """
        Get the current state ID.

        Returns:
            int: The current state ID.
        """
        return self._board_to_int(self.board)

    def reset(self) -> int:
        """
        Reset the environment to its initial state.

        Returns:
            int: The initial state ID.
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.score_value = 0
        return self.state_id()

    def display(self) -> None:
        """
        Display the current state of the environment.
        """
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))

    def is_forbidden(self, action: int) -> int:
        """
        Check if an action is forbidden in the current state.

        Args:
            action (int): The action to check.

        Returns:
            int: 0 if the action is allowed, 1 if the action is forbidden.
        """
        row, col = divmod(action, 3)
        return int(self.board[row, col] != 0)

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
        return np.where(self.board.flatten() == 0)[0]

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[int, float, bool, dict]: The next state ID, reward, done flag, and additional info.
        """
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
        """
        Get the current score value.

        Returns:
            float: The current score value.
        """
        return self.score_value

    @staticmethod
    def from_random_state() -> "TicTacToe":
        """
        Create a TicTacToe environment with a random state.

        Returns:
            TicTacToe: A new TicTacToe environment with a random state.
        """
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
        """
        Check if a player has won the game.

        Args:
            board (np.ndarray): The current board state.

        Returns:
            bool: True if a player has won, False otherwise.
        """
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
        """
        Convert an integer state representation to a board state.

        Args:
            s (int): The integer state representation.

        Returns:
            np.ndarray: The board state.
        """
        return np.array([(s // (3**i)) % 3 for i in range(9)]).reshape(3, 3)

    def _board_to_int(self, board: np.ndarray) -> int:
        """
        Convert a board state to an integer state representation.

        Args:
            board (np.ndarray): The board state.

        Returns:
            int: The integer state representation.
        """
        return sum(3**i * v for i, v in enumerate(board.flatten()))

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the game as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        return self.board.flatten()

    def action_vector(self, action: int) -> np.ndarray:
        """
        Get the vector encoding of the given action.

        Args:
            action (int): The action to encode.

        Returns:
            np.ndarray: The vector encoding of the action.
        """
        action_vector = np.zeros(9, dtype=int)
        action_vector[action] = 1
        return action_vector