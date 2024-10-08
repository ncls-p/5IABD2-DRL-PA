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
    env.reset()
    env.state = (0, 0)
    env.goal = (2, 2)
    # Test moving right
    assert env.p((0, 0), 1, (1, 0), 0) == 1.0
    assert env.p((0, 0), 1, (0, 0), 1) == 0.0
    # Test reaching goal
    assert env.p((1, 2), 1, (2, 2), 2) == 1.0
    assert env.p((2, 2), 1, (2, 2), 1) == 1.0
    # Test invalid move
    assert env.p((2, 0), 1, (2, 0), 1) == 1.0


def test_reset():
    env = GridWorld()
    env.state = (2, 2)
    env.done = True
    env.score_value = 10
    env.reset()
    assert env.state == (0, 0)
    assert not env.done
    assert env.score_value == 0


def test_available_actions():
    env = GridWorld()
    env.state = (0, 0)
    np.testing.assert_array_equal(env.available_actions(), np.array([1, 2]))
    env.state = (1, 1)
    np.testing.assert_array_equal(env.available_actions(), np.array([0, 1, 2, 3]))


def test_step():
    env = GridWorld(size=3)
    env.reset()
    env.state = (0, 0)
    env.goal = (2, 2)

    # Test moving right
    state, reward, done, _ = env.step(1)
    assert state == env.state_id()
    assert env.state == (1, 0)
    assert reward == -1.0
    assert not done
    assert env.score_value == -1.0

    # Test reaching goal
    env.state = (1, 2)
    state, reward, done, _ = env.step(1)
    assert env.state == (2, 2)
    assert reward == 1.0
    assert done
    assert env.score_value == -1.0 + 1.0

    # Test action after game is over
    state, reward, done, _ = env.step(1)
    assert env.state == (2, 2)
    assert reward == 0.0
    assert done
    assert env.score_value == 0.0

def test_from_random_state():
    env = GridWorld.from_random_state()
    assert 3 <= env.size <= 5
    assert 0 <= env.state[0] < env.size
    assert 0 <= env.state[1] < env.size
    assert 0 <= env.goal[0] < env.size
    assert 0 <= env.goal[1] < env.size
    assert env.state != env.goal