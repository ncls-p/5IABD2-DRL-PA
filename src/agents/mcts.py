import logging
import math
import random
from typing import Any, List, Optional
import numpy as np
from copy import deepcopy
from ..environments.farkle import Farkle
from src.environments import Environment
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCTSNode:
    def __init__(
        self,
        state: Any,
        action: Optional[int] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.action = action  # Action that led to this state
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[int] = []

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = 1.414) -> "MCTSNode":
        if not self.children:
            raise ValueError("Node has no children")

        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                exploitation = 0
                exploration = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
            choices_weights.append(exploitation + exploration)

        best_child_node = self.children[np.argmax(choices_weights)]
        logger.debug(f"Best child selected with action: {best_child_node.action}")
        return best_child_node

    def add_child(self, state: Any, action: int) -> "MCTSNode":
        child = MCTSNode(state=state, action=action, parent=self)
        self.children.append(child)
        return child


class MCTSAgent:
    def __init__(
        self, env: Environment, num_simulations: int = 1000, max_depth: int = 50
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.player = 1

    def copy_environment(self, env: Environment) -> Environment:
        return deepcopy(env)

    def simulate(self, env: Environment, max_steps: int = 50) -> float:
        steps = 0
        while not env.is_game_over() and steps < max_steps:
            action = random.choice(env.available_actions())
            _, reward, done, _ = env.step(action)
            if done:  # Early stopping on game completion
                break
            steps += 1
        return env.score()

    def choose_action(self, state: Any, temperature: float = 0.0) -> int:
        root = MCTSNode(state)
        root.untried_actions = list(self.env.available_actions())
        logger.debug(f"Choosing action for state: {state}")

        for i in range(self.num_simulations):
            node = root
            sim_env = self.copy_environment(self.env)

            # Selection
            depth = 0
            while (
                not sim_env.is_game_over()
                and node.is_fully_expanded()
                and depth < self.max_depth
            ):
                node = node.best_child()
                if node.action is not None:
                    sim_env.step(node.action)
                depth += 1

            # Expansion
            if not sim_env.is_game_over() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                next_state, _, done, _ = sim_env.step(action)
                node = node.add_child(next_state, action)
                node.untried_actions = list(sim_env.available_actions())

            # Simulation
            simulation_result = self.simulate(sim_env)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += simulation_result
                node = node.parent

            if (i + 1) % 100 == 0:
                logger.debug(f"Completed {i + 1} simulations")

        # Select best action from root using temperature
        if temperature == 0.0:
            best_child = root.best_child(c_param=0.0)
        else:
            children_values = np.array(
                [c.value / c.visits if c.visits > 0 else 0 for c in root.children],
                dtype=np.float32,
            )
            probabilities = np.exp(children_values / temperature)
            probabilities = probabilities / np.sum(probabilities)
            best_child = root.children[
                np.random.choice(len(root.children), p=probabilities)
            ]

        if best_child.action is None:
            # Fallback to random action if no valid action found
            available_actions = list(self.env.available_actions())
            if not available_actions:
                raise ValueError("No available actions")
            return random.choice(available_actions)
        # In choose_action()
        logger.debug(
            f"Selected action {best_child.action} with value {best_child.value/best_child.visits}"
        )
        return best_child.action

    def train(self, num_episodes: int = 1000) -> List[float]:
        scores = []
        steps_per_episode = []
        action_times = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            steps = 0
            episode_action_times = []
            
            while not done:
                start_time = time.time()
                action = self.choose_action(state)
                episode_action_times.append(time.time() - start_time)
                
                state, reward, done, info = self.env.step(action)
                steps += 1
                if isinstance(self.env, Farkle):
                    self.player = info["current_player"]

            episode_score = self.env.score()
            if isinstance(episode_score, list):
                scores.append(episode_score[self.player - 1])
            else:
                scores.append(episode_score)
            
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))

            logger.info(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Score: {np.mean(scores[-100:]):.2f}, "
                f"Steps: {steps}, "
                f"Avg Action Time: {np.mean(episode_action_times):.4f}s"
            )

        return scores, steps_per_episode, action_times
