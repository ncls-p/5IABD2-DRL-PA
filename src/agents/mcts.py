import logging
import math
import random
from typing import Any, List

import numpy as np

from ..environments.farkle import Farkle
from src.environments import Environment

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCTSNode:
    def __init__(self, state: Any, parent=None):
        self.state = state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[int] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> "MCTSNode":
        choices_weights = [
            (c.value / c.visits)
            + c_param * math.sqrt((2 * math.log(self.visits) / c.visits))
            for c in self.children
        ]
        best_child_node = self.children[np.argmax(choices_weights)]
        logger.debug(f"Best child selected with state: {best_child_node.state}")
        return best_child_node

    def rollout_policy(self, env: Environment) -> int:
        action = random.choice(env.available_actions())
        logger.debug(f"Rollout policy selected action: {action}")
        return action


class MCTSAgent:
    def __init__(self, env: Environment, num_simulations: int = 1000):
        self.env = env
        self.num_simulations = num_simulations
        self.player = 1  # Initialize the player

    def choose_action(self, state: Any) -> int:
        root = MCTSNode(state)
        root.untried_actions = list(self.env.available_actions())
        logger.debug(f"Choosing action for state: {state}")

        for _ in range(self.num_simulations):
            node = root
            sim_env = self.env.__class__()  # Create a copy of the environment
            sim_env.reset()
            sim_env_state = sim_env.state_id()

            # Synchronize the simulated environment state with the current state
            while sim_env_state != state:
                action = sim_env.available_actions()[0]
                sim_env_state, _, _, _ = sim_env.step(action)

            # Selection
            while not sim_env.is_game_over() and node.is_fully_expanded():
                node = node.best_child()
                sim_env_state, _, done, _ = sim_env.step(node.state)

            # Expansion
            if not sim_env.is_game_over():
                action = node.untried_actions.pop()
                sim_env_state, _, done, _ = sim_env.step(action)
                child = MCTSNode(sim_env_state, parent=node)
                child.untried_actions = list(sim_env.available_actions())
                node.children.append(child)
                node = child
                logger.debug(f"Expanded node with state: {child.state}")

            # Simulation
            while not sim_env.is_game_over():
                action = node.rollout_policy(sim_env)
                sim_env_state, _, done, _ = sim_env.step(action)
                node = MCTSNode(sim_env_state, parent=node)

            # Backpropagation
            while node is not None:
                node.visits += 1
                scores = sim_env.score()
                if isinstance(scores, list):
                    node.value += scores[
                        self.player - 1
                    ]  # Use the score of the current player
                else:
                    node.value += scores
                logger.debug(
                    f"Backpropagating node with state: {node.state}, visits: {node.visits}, value: {node.value}"
                )
                node = node.parent

        return root.best_child(c_param=0.0).state

    def train(self, num_episodes: int = 1000) -> List[float]:
        scores = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            while not done:
                action = self.choose_action(state)
                state, reward, done, info = self.env.step(action)
                if isinstance(self.env, Farkle):
                    self.player = info["current_player"]  # Update the current player

            episode_score = self.env.score()
            if isinstance(episode_score, list):
                scores.append(episode_score[self.player - 1])
            else:
                scores.append(episode_score)

            if (episode + 1) % 100 == 0:
                logger.info(
                    f"Episode {episode + 1}/{num_episodes}, Avg Score: {np.mean(scores[-100:]):.2f}"
                )

        return scores
