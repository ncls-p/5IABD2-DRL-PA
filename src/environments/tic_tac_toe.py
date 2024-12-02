from typing import Optional, Tuple

import numpy as np

from src.environments import Environment


class TicTacToe(Environment):
    def __init__(self):
        """Initialize the TicTacToe environment."""
        self.board: np.ndarray = np.zeros((3, 3), dtype=int)
        self.current_player: int = 1
        self.done: bool = False
        self.winner: Optional[int] = (
            None  # 1 for player 1, 2 for player 2, None for no winner
        )
    
    def env_name(self):
        return "tic_tac_toe"

    def render(self) -> None:
        """Render the current state of the environment."""
        self.display()

    def num_states(self) -> int:
        """Get the number of states in the environment."""
        return 3**9

    def num_actions(self) -> int:
        """Get the number of possible actions in the environment."""
        return 9

    def num_rewards(self) -> int:
        """Get the number of possible rewards in the environment."""
        return 3

    def reward(self, i: int) -> float:
        """Get the reward value for a given reward index."""
        rewards = [0.0, 0.5, 1.0]  # Draw, Loss, Win
        return rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """Get the transition probability for a given state, action, next state, and reward index."""
        board = self._int_to_board(s)
        next_board = self._int_to_board(s_p)
        row, col = divmod(a, 3)

        if board[row, col] != 0:
            # Invalid move
            return 1.0 if s_p == s and r_index == 0 else 0.0

        board[row, col] = self.current_player
        if np.array_equal(board, next_board):
            if self._check_win(board, self.current_player):
                return 1.0 if r_index == 2 else 0.0  # Win
            elif np.all(board != 0):
                return 1.0 if r_index == 0 else 0.0  # Draw
            else:
                return 1.0 if r_index == 0 else 0.0  # Ongoing game
        return 0.0

    def state_id(self) -> int:
        """Get the current state ID."""
        return self._board_to_int(self.board)

    def reset(self) -> int:
        """Reset the environment to its initial state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.state_id()

    def display(self) -> None:
        """Display the current state of the environment."""
        symbols = {0: ".", 1: "X", 2: "O"}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))

    def is_forbidden(self, action: int) -> int:
        """Check if an action is forbidden in the current state."""
        row, col = divmod(action, 3)
        return int(self.board[row, col] != 0)

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.done

    def available_actions(self) -> np.ndarray:
        """Get the available actions in the current state."""
        return np.where(self.board.flatten() == 0)[0]

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take a step in the environment based on the given action."""
        if self.done:
            return self.state_id(), 0.0, True, {}

        if self.is_forbidden(action):
            # Invalid action
            return self.state_id(), 0.0, False, {}

        row, col = divmod(action, 3)
        self.board[row, col] = self.current_player

        if self._check_win(self.board, self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.board != 0):
            self.done = True
            self.winner = None
            reward = 0.0  # Draw
        else:
            reward = 0.0
            self.current_player = 3 - self.current_player  # Switch player

        return (
            self.state_id(),
            reward,
            self.done,
            {"current_player": self.current_player, "winner": self.winner},
        )

    def _check_win(self, board: np.ndarray, player: int) -> bool:
        """Check if a player has won the game."""
        return bool(
            np.any(np.all(board == player, axis=0))
            or np.any(np.all(board == player, axis=1))
            or np.all(np.diag(board) == player)
            or np.all(np.diag(np.fliplr(board)) == player)
        )

    def _int_to_board(self, s: int) -> np.ndarray:
        """Convert an integer state representation to a board state."""
        return np.array([(s // (3**i)) % 3 for i in range(9)]).reshape(3, 3)

    def _board_to_int(self, board: np.ndarray) -> int:
        """Convert a board state to an integer state representation."""
        return int(sum((board.flatten()[i] * (3**i) for i in range(9))))

    def score(self) -> float:
        """Get the current score value.

        Returns:
            float: The current score value.
        """
        if self.winner == 1:
            return 1.0
        elif self.winner == 2:
            return -1.0
        else:
            return 0.0

    def state_vector(self) -> np.ndarray:
        """Get the current state of the game as a vector encoding."""
        return self.board.flatten()

    def action_vector(self, action: int) -> np.ndarray:
        """Get the vector encoding of the given action."""
        action_vector = np.zeros(9, dtype=int)
        action_vector[action] = 1
        return action_vector

    @staticmethod
    def from_random_state() -> "TicTacToe":
        """
        Create a TicTacToe environment with a random state.

        Returns:
            TicTacToe: A new TicTacToe environment with a random state.
        """
        env = TicTacToe()
        num_moves = np.random.randint(0, 5)  # Limit moves to 0 to 4
        players = [1, 2]
        current_player = np.random.choice(players)
        env.current_player = current_player

        # Simulate random moves up to num_moves
        for _ in range(num_moves):
            available_actions = env.available_actions()
            if len(available_actions) == 0:
                break
            action = np.random.choice(available_actions)
            env.step(action)
            if env.done:
                break  # Stop if the game is over

        # Set the current player for the next move
        if not env.done:
            env.current_player = 3 - env.current_player

        return env