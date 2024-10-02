import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.farkle import Farkle


def test_initialization():
    env = Farkle()
    assert env.target_score == 10000
    assert len(env.dice) == 2
    assert len(env.dice[0]) == 6
    assert len(env.dice[1]) == 6
    assert env.current_player == 1
    assert env.scores == [0, 0]
    assert not env.done
    assert not env.final_round
    assert env.final_round_starter is None


def test_reset():
    env = Farkle()
    env.scores = [5000, 6000]
    env.current_player = 2
    env.done = True
    env.final_round = True
    env.final_round_starter = 1

    initial_state = env.reset()
    assert 0 <= initial_state < 6**6
    assert env.scores == [0, 0]
    assert env.current_player == 1
    assert not env.done
    assert not env.final_round
    assert env.final_round_starter is None


def test_step_bank_points():
    env = Farkle()
    env.turn_score = 350
    _, reward, done, info = env.step(0)  # Bank points
    assert env.scores[0] == 350
    assert reward == 350 / 10000
    assert not done
    assert info["turn_ended"]
    assert env.current_player == 2


def test_step_keep_dice():
    env = Farkle()
    env.dice[0] = np.array([1, 2, 3, 4, 5, 6])
    # Corrected action to keep indices 0 and 4 (values 1 and 5)
    _, reward, done, info = env.step(17)  # Action 17: binary 010001
    assert env.turn_score == 150  # 1 (100 points) + 5 (50 points)
    assert reward == 0.015  # 150 / 10000
    assert not done
    assert not info["turn_ended"]
    assert env.current_player == 1


def test_step_farkle():
    env = Farkle()
    env.dice[0] = np.array([2, 3, 4, 2, 3, 4])
    _, reward, done, info = env.step(63)  # Keep all dice
    assert env.turn_score == 1500  # Three pairs worth 1500 points
    assert reward == 0.15  # 1500 / 10000
    assert not done
    assert not info["turn_ended"]
    assert env.current_player == 1

    # Test an invalid keep (no scoring dice)
    env.dice[0] = np.array([2, 3, 4, 6, 2, 3])
    _, reward, done, info = env.step(
        1
    )  # Try to keep first die (value 2, which doesn't score alone)
    assert env.turn_score == 0
    assert reward == -1
    assert not done
    assert info["turn_ended"]
    assert env.current_player == 2  # Turn switches to player 2

    # Update dice for the new current player (player 2)
    env.dice[1] = np.array([1, 1, 1, 2, 3, 4])
    _, reward, done, info = env.step(7)  # Keep first three dice (three 1s)
    assert env.turn_score == 1000
    assert reward == 0.1  # 1000 / 10000
    assert not done
    assert not info["turn_ended"]
    assert env.current_player == 2


def test_final_round():
    env = Farkle()
    env.scores = [9900, 9800]
    env.turn_score = 150
    env.step(0)  # Bank points to trigger final round
    assert env.final_round
    assert env.final_round_starter == 1

    env.turn_score = 200
    _, _, done, info = env.step(0)  # Bank points for second player
    assert done
    assert info["winner"] == 1


def test_calculate_score():
    env = Farkle()

    # Test scoring for five 1s
    env.dice[0] = np.array([1, 1, 1, 1, 1, 2])
    score = env._calculate_score(np.array([1, 1, 1, 1, 1, 0]))
    assert score == 4000  # Five 1s: 1000 (three 1s) doubled twice: 1000 * 4 = 4000

    # Test scoring for three 1s and two 5s
    env.dice[0] = np.array([1, 1, 1, 5, 5, 2])
    score = env._calculate_score(np.array([1, 1, 1, 1, 1, 0]))
    assert score == 1100  # Three 1s (1000) + two 5s (50 each) = 1100

    # Test scoring for two triplets
    env.dice[0] = np.array([2, 2, 2, 3, 3, 3])
    score = env._calculate_score(np.array([1, 1, 1, 1, 1, 1]))
    assert score == 500  # Three 2s (200) + three 3s (300) = 500

    # Test scoring for a straight
    env.dice[0] = np.array([1, 2, 3, 4, 5, 6])
    score = env._calculate_score(np.array([1, 1, 1, 1, 1, 1]))
    assert score == 1500  # Straight scores 1500 points


def test_is_farkle():
    env = Farkle()
    env.dice[0] = np.array([2, 3, 4, 6, 2, 3])
    assert env._is_farkle()

    env.dice[0] = np.array([1, 3, 4, 6, 2, 3])
    assert not env._is_farkle()


def test_available_actions():
    env = Farkle()
    env.dice[0] = np.array([1, 1, 2, 3, 4, 5])
    actions = env.available_actions()
    assert 0 in actions  # Always available (bank)
    assert 1 in actions  # Keep first die
    assert 3 in actions  # Keep first two dice
    assert 17 in actions  # Keep first and fifth dice (indices 0 and 4)


def test_is_game_over():
    env = Farkle()
    assert not env.is_game_over()
    env.done = True
    assert env.is_game_over()


if __name__ == "__main__":
    pytest.main()
