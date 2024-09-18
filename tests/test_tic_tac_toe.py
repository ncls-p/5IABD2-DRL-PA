import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.tic_tac_toe import TicTacToe


def test_initialization():
    env = TicTacToe()
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert not env.done
    assert env.score_value == 0


def test_num_states_actions_rewards():
    env = TicTacToe()
    assert env.num_states() == 3**9
    assert env.num_actions() == 9
    assert env.num_rewards() == 3


def test_reward():
    env = TicTacToe()
    assert env.reward(0) == -1.0
    assert env.reward(1) == 0.0
    assert env.reward(2) == 1.0


def test_p():
    env = TicTacToe()
    initial_state = env.state_id()

    # Test making a valid move
    assert (
        env.p(
            initial_state,
            0,
            env._board_to_int(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])),
            0,
        )
        == 1.0
    )

    # Test winning move
    winning_state = env._board_to_int(np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]))
    assert (
        env.p(
            winning_state,
            2,
            env._board_to_int(np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])),
            2,
        )
        == 1.0
    )

    # Test invalid move (trying to place in an occupied cell)
    occupied_state = env._board_to_int(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    assert env.p(occupied_state, 0, occupied_state, 1) == 1.0

    # Test valid move that doesn't result in a win
    assert (
        env.p(
            initial_state,
            4,
            env._board_to_int(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
            0,
        )
        == 1.0
    )


def test_reset():
    env = TicTacToe()
    env.board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    env.current_player = 2
    env.done = True
    env.score_value = 10
    assert env.reset() == 0
    assert np.all(env.board == 0)
    assert env.current_player == 1
    assert not env.done
    assert env.score_value == 0


def test_available_actions():
    env = TicTacToe()
    env.board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    np.testing.assert_array_equal(env.available_actions(), np.array([2, 3, 6, 7, 8]))


def test_step():
    env = TicTacToe()

    # Test making a move
    state, reward, done, _ = env.step(0)
    assert state == 1
    assert reward == 0
    assert not done
    assert env.current_player == 2

    # Test winning move
    env.board = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    env.current_player = 1
    state, reward, done, _ = env.step(2)
    assert reward == 1
    assert done
    assert env.score_value == 1

    # Test action after game is over
    state, reward, done, _ = env.step(3)
    assert state == env._board_to_int(np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]))
    assert reward == 0
    assert done
    assert env.score_value == 1


def test_from_random_state():
    env = TicTacToe.from_random_state()
    assert 0 <= np.sum(env.board != 0) <= 4
    assert env.current_player in [1, 2]


def test_check_win():
    env = TicTacToe()
    # Test row win
    env.board = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert env._check_win(env.board)
    # Test column win
    env.board = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    assert env._check_win(env.board)
    # Test diagonal win
    env.board = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert env._check_win(env.board)
    # Test no win
    env.board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    assert not env._check_win(env.board)


def test_int_to_board_and_board_to_int():
    env = TicTacToe()
    board = np.array([[1, 2, 0], [0, 1, 2], [0, 0, 0]])
    board_int = env._board_to_int(board)
    assert np.array_equal(env._int_to_board(board_int), board)
