import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.line_world import LineWorld


def test_initialization():
    env = LineWorld(size=7)
    assert env.size == 7
    assert env.state == 0
    assert env.target_position == 6
    assert not env.done
    assert env.score_value == 0


def test_num_states_actions_rewards():
    env = LineWorld(size=5)
    assert env.num_states() == 5
    assert env.num_actions() == 2
    assert env.num_rewards() == 3


def test_reward():
    env = LineWorld()
    assert env.reward(0) == -1.0
    assert env.reward(1) == 0.0
    assert env.reward(2) == 1.0


def test_p():
    env = LineWorld(size=5)
    env.reset()
    env.state = 0
    env.target_position = 4
    # Test moving right
    assert env.p(0, 1, 1, 0) == 1.0
    # Test invalid move (left from state 0)
    assert env.p(0, 0, 0, 1) == 1.0
    # Test reaching target
    assert env.p(3, 1, 4, 2) == 1.0


def test_reset():
    env = LineWorld()
    env.state = 3
    env.done = True
    env.score_value = 10
    env.reset()
    assert env.state == 0
    assert not env.done
    assert env.score_value == 0


def test_available_actions():
    env = LineWorld()
    env.state = 0
    np.testing.assert_array_equal(env.available_actions(), np.array([1]))
    env.state = 3
    np.testing.assert_array_equal(env.available_actions(), np.array([0, 1]))


def test_step():
    env = LineWorld(size=5)
    env.reset()
    env.state = 0
    env.target_position = 4

    # Test moving right
    state, reward, done, _ = env.step(1)
    assert state == 1
    assert reward == -1.0
    assert not done
    assert env.score_value == -1.0

    # Test reaching target
    env.state = 3
    state, reward, done, _ = env.step(1)
    assert state == 4
    assert reward == 1.0
    assert done
    assert env.score_value == -1.0 + 1.0

    # Test action after game is over
    state, reward, done, _ = env.step(1)
    assert state == 4
    assert reward == 0.0
    assert done
    assert env.score_value == 0.0

def test_from_random_state():
    env = LineWorld.from_random_state()
    assert 5 <= env.size <= 10
    assert 0 <= env.state < env.size
    assert 0 <= env.target_position < env.size
    assert env.state != env.target_position