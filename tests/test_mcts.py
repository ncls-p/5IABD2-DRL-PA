import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.agents.mcts import MCTSAgent, MCTSNode
from src.environments import Environment


class MockEnvironment(Environment):
    def __init__(self):
        self.state = 0
        self.done = False

    def reset(self):
        self.state = 0
        self.done = False
        return self.state

    def step(self, action):
        self.state += 1
        reward = 1 if self.state == 5 else 0
        self.done = self.state == 5
        return self.state, reward, self.done, {}

    def available_actions(self):
        return [0, 1]

    def is_game_over(self):
        return self.done

    def state_id(self):
        return self.state

    def score(self):
        return self.state

    def num_states(self):
        return 6  # 0 à 5

    def num_actions(self):
        return 2  # 0 et 1

    def num_rewards(self):
        return 2  # 0 et 1

    def reward(self, r):
        return r

    def p(self, s, a, s_next, r):
        return 1.0

    def render(self):
        print(f"Current state: {self.state}")

    def is_forbidden(self, state, action):
        return False


def test_mcts_node_initialization():
    node = MCTSNode(state=0)
    assert node.state == 0
    assert node.parent is None
    assert node.children == []
    assert node.visits == 0
    assert node.value == 0.0
    assert node.untried_actions == []


def test_mcts_node_is_fully_expanded():
    node = MCTSNode(state=0)
    assert node.is_fully_expanded()
    node.untried_actions = [1, 2]
    assert not node.is_fully_expanded()


def test_mcts_node_best_child():
    parent = MCTSNode(state=0)
    child1 = MCTSNode(state=1, parent=parent)
    child2 = MCTSNode(state=2, parent=parent)
    parent.children = [child1, child2]

    child1.visits = 10
    child1.value = 5
    child2.visits = 5
    child2.value = 4

    parent.visits = 15

    best_child = parent.best_child()
    assert best_child == child2


def test_mcts_node_rollout_policy():
    env = MockEnvironment()
    node = MCTSNode(state=0)
    action = node.rollout_policy(env)
    assert action in [0, 1]


def test_mcts_agent_initialization():
    env = MockEnvironment()
    agent = MCTSAgent(env, num_simulations=100)
    assert agent.env == env
    assert agent.num_simulations == 100
    assert agent.player == 1


def test_mcts_agent_choose_action():
    env = MockEnvironment()
    agent = MCTSAgent(env, num_simulations=100)
    action = agent.choose_action(0)
    assert isinstance(action, int)
    assert 0 <= action <= 5


def test_mcts_agent_train():
    env = MockEnvironment()
    agent = MCTSAgent(env, num_simulations=10)
    scores = agent.train(num_episodes=5)
    assert len(scores) == 5
    assert all(isinstance(score, (int, float)) for score in scores)


if __name__ == "__main__":
    pytest.main()
