import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.line_world import LineWorld


def test_initialization():
    env = LineWorld(size=7)
    assert env.size == 7
    assert env.state == 0
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
    # Test left edge
    assert env.p(0, 0, 0, 1) == 1.0
    assert env.p(0, 0, 1, 0) == 0.0
    # Test right edge
    assert env.p(4, 1, 4, 1) == 1.0
    assert env.p(4, 1, 3, 0) == 0.0
    # Test moving left
    assert env.p(2, 0, 1, 0) == 1.0
    assert env.p(2, 0, 1, 1) == 0.0
    # Test moving right
    assert env.p(2, 1, 3, 0) == 1.0
    assert env.p(2, 1, 3, 1) == 0.0
    # Test reaching goal
    assert env.p(3, 1, 4, 2) == 1.0
    assert env.p(3, 1, 4, 0) == 0.0


def test_reset():
    env = LineWorld()
    env.state = 3
    env.done = True
    env.score_value = 10
    assert env.reset() == 0
    assert env.state == 0
    assert not env.done
    assert env.score_value == 0


def test_available_actions():
    env = LineWorld()
    np.testing.assert_array_equal(env.available_actions(), np.array([0, 1]))


def test_step():
    env = LineWorld(size=5)

    # Test moving right
    state, reward, done, _ = env.step(1)
    assert state == 1
    assert reward == -1
    assert not done
    assert env.score_value == -1

    # Test moving left
    state, reward, done, _ = env.step(0)
    assert state == 0
    assert reward == -1
    assert not done
    assert env.score_value == -2

    # Test reaching goal
    env.state = 3
    state, reward, done, _ = env.step(1)
    assert state == 4
    assert reward == 1
    assert done
    assert env.score_value == -1

    # Test action after game is over
    state, reward, done, _ = env.step(1)
    assert state == 4
    assert reward == 0
    assert done
    assert env.score_value == -1


def test_from_random_state():
    env = LineWorld.from_random_state()
    assert 5 <= env.size <= 10
    assert 0 <= env.state < env.size
