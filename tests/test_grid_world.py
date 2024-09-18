import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.grid_world import GridWorld


def test_initialization():
    env = GridWorld(size=4)
    assert env.size == 4
    assert env.state == (0, 0)
    assert env.goal == (3, 3)
    assert not env.done
    assert env.score_value == 0


def test_num_states_actions_rewards():
    env = GridWorld(size=3)
    assert env.num_states() == 9
    assert env.num_actions() == 4
    assert env.num_rewards() == 3


def test_reward():
    env = GridWorld()
    assert env.reward(0) == -1.0
    assert env.reward(1) == 0.0
    assert env.reward(2) == 1.0


def test_p():
    env = GridWorld(size=3)
    # Test moving right
    assert env.p((0, 0), 1, (1, 0), 0) == 1.0
    assert env.p((0, 0), 1, (1, 0), 1) == 0.0
    # Test reaching goal
    assert env.p((2, 2), 1, (2, 2), 1) == 1.0
    assert env.p((1, 2), 1, (2, 2), 2) == 1.0
    # Test invalid move
    assert env.p((0, 0), 3, (0, 0), 1) == 1.0


def test_reset():
    env = GridWorld()
    env.set_state((2, 2))
    env.done = True
    env.score_value = 10
    assert env.reset() == 0
    assert env.state == (0, 0)
    assert not env.done
    assert env.score_value == 0


def test_available_actions():
    env = GridWorld()
    np.testing.assert_array_equal(env.available_actions(), np.array([0, 1, 2, 3]))


def test_step():
    env = GridWorld(size=3)

    # Test moving right
    state, reward, done, _ = env.step(1)
    assert state == 1
    assert reward == -1
    assert not done
    assert env.score_value == -1

    # Test reaching goal
    env.set_state((2, 2))
    state, reward, done, _ = env.step(1)
    assert state == 8
    assert reward == 1
    assert done
    assert env.score_value == 0

    # Test action after game is over
    state, reward, done, _ = env.step(1)
    assert state == 8
    assert reward == 0
    assert done
    assert env.score_value == 0


def test_from_random_state():
    env = GridWorld.from_random_state()
    assert 3 <= env.size <= 5
    assert 0 <= env.state[0] < env.size
    assert 0 <= env.state[1] < env.size
