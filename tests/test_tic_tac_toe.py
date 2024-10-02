import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.tic_tac_toe import TicTacToe


def test_initialization():
    env = TicTacToe()
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert not env.done
    assert env.winner is None


def test_num_states_actions_rewards():
    env = TicTacToe()
    assert env.num_states() == 3**9
    assert env.num_actions() == 9
    assert env.num_rewards() == 3


def test_reward():
    env = TicTacToe()
    assert env.reward(0) == 0.0  # Draw or ongoing game
    assert env.reward(1) == 0.5  # Loss (not used in this implementation)
    assert env.reward(2) == 1.0  # Win


def test_p():
    env = TicTacToe()
    initial_state = env.state_id()

    # Test making a valid move
    next_board = env._int_to_board(initial_state).copy()
    next_board[0, 0] = env.current_player
    next_state = env._board_to_int(next_board)
    assert env.p(initial_state, 0, next_state, 0) == 1.0  # Ongoing game

    # Test winning move
    env.board = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    env.current_player = 1
    winning_board = env.board.copy()
    winning_board[0, 2] = env.current_player
    winning_state = env._board_to_int(winning_board)
    assert env.p(env.state_id(), 2, winning_state, 2) == 1.0  # Win

    # Test invalid move (cell already occupied)
    env.board = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    occupied_state = env.state_id()
    assert env.p(occupied_state, 0, occupied_state, 0) == 1.0  # No change in state


def test_reset():
    env = TicTacToe()
    env.board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]])
    env.current_player = 2
    env.done = True
    env.winner = 1
    env.reset()
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert not env.done
    assert env.winner is None


def test_available_actions():
    env = TicTacToe()
    env.board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    expected_actions = np.array([2, 3, 6, 7, 8])
    np.testing.assert_array_equal(env.available_actions(), expected_actions)


def test_step():
    env = TicTacToe()
    env.current_player = 1

    # Test making a valid move
    state, reward, done, info = env.step(0)
    assert state == env.state_id()
    assert env.board[0, 0] == 1
    assert reward == 0.0  # No win yet
    assert not done
    assert env.current_player == 2  # Player switched

    # Test invalid move (cell already occupied)
    state, reward, done, info = env.step(0)
    assert state == env.state_id()
    assert reward == 0.0  # No reward change
    assert not done
    assert env.current_player == 2  # Player should not switch

    # Set up a winning scenario for player 2
    env.board = np.array([[1, 1, 0], [2, 2, 0], [0, 0, 0]])
    env.current_player = 2
    state, reward, done, info = env.step(5)  # Player 2 plays position 5
    assert env.board[1, 2] == 2
    assert reward == 1.0  # Player 2 wins
    assert done
    assert env.winner == 2

    # Check the score
    assert (
        env.score() == -1.0
    )  # Player 2 wins, so score is -1.0 for player 1, 1.0 for player 2

    # Test action after game is over
    state, reward, done, info = env.step(6)
    assert state == env.state_id()
    assert reward == 0.0
    assert done  # Game remains over
    assert env.winner == 2


def test_score():
    env = TicTacToe()
    # Ongoing game
    assert env.score() == 0.0

    # Player 1 wins
    env.winner = 1
    assert env.score() == 1.0

    # Player 2 wins
    env.winner = 2
    assert env.score() == -1.0

    # Draw
    env.winner = None
    assert env.score() == 0.0


def test_from_random_state():
    env = TicTacToe.from_random_state()
    num_moves = np.count_nonzero(env.board)
    assert 0 <= num_moves <= 8
    assert env.current_player in [1, 2]


def test_check_win():
    env = TicTacToe()
    # Test row win
    env.board = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert env._check_win(env.board, 1)
    # Test column win
    env.board = np.array([[2, 0, 0], [2, 0, 0], [2, 0, 0]])
    assert env._check_win(env.board, 2)
    # Test diagonal win
    env.board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert env._check_win(env.board, 1)
    # Test anti-diagonal win
    env.board = np.array([[0, 0, 2], [0, 2, 0], [2, 0, 0]])
    assert env._check_win(env.board, 2)
    # Test no win
    env.board = np.array([[1, 2, 1], [2, 1, 2], [2, 1, 2]])
    assert not env._check_win(env.board, 1)
    assert not env._check_win(env.board, 2)


def test_int_to_board_and_board_to_int():
    env = TicTacToe()
    board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    board_int = env._board_to_int(board)
    assert np.array_equal(env._int_to_board(board_int), board)


if __name__ == "__main__":
    pytest.main()