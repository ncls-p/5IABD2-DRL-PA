import logging
import math
import time
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np

from src.environments import Environment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MCTSNode:
    def __init__(
        self,
        state_id: int,
        action: Optional[int] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state_id = state_id  # Changed from state to state_id
        self.action = action
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
        child = MCTSNode(state_id=state, action=action, parent=self)
        self.children.append(child)
        return child


class MCTSAgent:
    def __init__(
        self,
        env: Environment,
        num_simulations: int = 1000,
        max_depth: int = 50,
        exploration_weight: float = 2.0,
        simulation_temp: float = 1.5,
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.player = 1
        self.exploration_weight = exploration_weight
        self.simulation_temp = simulation_temp
        self._available_actions: np.ndarray = np.array(
            [], dtype=int
        )  # Initialize as empty array
        self._last_state = None
        self._value_stats = {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "recent_values": [],
            "best_value": float("-inf"),
        }
        self._sim_cache = {}
        self._cache_hits = 0
        self._total_sims = 0
        self._episode_count = 0
        self._min_exploration = 0.4
        self._recent_window = 100
        self._last_action = None
        self._consecutive_failures = 0
        self.initial_epsilon = 1.0
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.1
        self.current_epsilon = self.initial_epsilon

    def copy_environment(self, env: Environment) -> Environment:
        return deepcopy(env)

    def get_available_actions(self, state: Any) -> np.ndarray:
        if state != self._last_state:
            self._available_actions = self.env.available_actions()
            self._last_state = state
        return self._available_actions

    def update_value_stats(self, value: float) -> None:
        self._value_stats['count'] += 1
        count = self._value_stats["count"]
        self._value_stats['min'] = min(self._value_stats['min'], value)
        self._value_stats['max'] = max(self._value_stats['max'], value)
        self._value_stats["sum"] += value
        delta = value - self._value_stats['mean']
        self._value_stats['mean'] += delta / count
        delta2 = value - self._value_stats['mean']
        self._value_stats['var'] += delta * delta2

    def normalize_value(self, value: float) -> float:
        if self._value_stats['count'] < 2:
            return value
        std = np.sqrt(self._value_stats['var'] / (self._value_stats['count'] - 1))
        if std == 0:
            z_score = 0
        else:
            z_score = (value - self._value_stats["mean"]) / std
        v_range = self._value_stats['max'] - self._value_stats['min']
        if v_range == 0:
            return z_score
        v_norm = (value - self._value_stats["min"]) / v_range
        return 0.5 * (z_score + v_norm)

    def simulate(self, env: Environment, max_steps: int = 50) -> float:
        # Use state_id() for hashing instead of env.state
        state_hash = env.state_id()
        self._total_sims += 1

        if state_hash in self._sim_cache:
            self._cache_hits += 1
            cached_value = self._sim_cache[state_hash]
            noise_scale = 1.0 / (1 + self._episode_count / 500.0)
            noise = np.random.uniform(0, noise_scale)
            return cached_value + noise

        steps = 0
        total_reward = 0
        discount = 0.99
        while not env.is_game_over() and steps < max_steps:
            available_actions = env.available_actions()
            if len(available_actions) == 0:
                break
            base_rate = max(self._min_exploration, 2.0 / (1.0 + steps / 10.0))
            if np.random.random() < base_rate:
                action = np.random.choice(available_actions)
            else:
                values = np.ones(len(available_actions))
                temp = max(0.5, self.simulation_temp / (1 + steps/10))
                probs = np.exp(values / temp)
                probs = probs / np.sum(probs)
                action = np.random.choice(available_actions, p=probs)
            _, reward, done, _ = env.step(action)
            total_reward += (discount**steps) * reward
            steps += 1
            if done:
                break
        if total_reward > self._value_stats['best_value']:
            self._value_stats["best_value"] = total_reward
        if steps > 0:
            self._sim_cache[state_hash] = total_reward
            if len(self._sim_cache) > 10000:
                self._sim_cache.clear()
        return total_reward

    def choose_action(self, state_id: int, temperature: float = 1.0) -> int:
        root = MCTSNode(state_id)  # Changed from state to state_id
        available_actions = self.get_available_actions(state_id)
        root.untried_actions = list(available_actions)

        if len(root.untried_actions) == 1:
            return root.untried_actions[0]
        for _ in range(self.num_simulations):
            node = root
            sim_env = self.copy_environment(self.env)
            path = [node]
            depth = 0
            while (
                not sim_env.is_game_over()
                and node.is_fully_expanded()
                and depth < self.max_depth
            ):
                total_visits = max(1, sum(c.visits for c in node.children))
                log_total = np.log(total_visits)
                values = []
                for child in node.children:
                    value = child.value / max(1, child.visits)
                    explore_term = self.exploration_weight * np.sqrt(
                        log_total / child.visits
                    )
                    values.append(value + explore_term)
                node = node.children[np.argmax(values)]
                path.append(node)
                if node.action is not None:
                    sim_env.step(node.action)
                depth += 1
            if not sim_env.is_game_over() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                next_state, _, done, _ = sim_env.step(action)
                node = node.add_child(next_state, action)
                node.untried_actions = list(sim_env.available_actions())
                path.append(node)
            value = self.simulate(sim_env)
            self.update_value_stats(value)
            decay = 0.95
            for i, node in enumerate(reversed(path)):
                node.visits += 1
                node.value += value * (decay**i)
        visit_threshold = max(2, int(np.sqrt(self.num_simulations)))
        qualified_children = [c for c in root.children if c.visits >= visit_threshold]

        # Handle case where no qualified children exist
        if not qualified_children and not root.untried_actions:
            # If no actions are available, return 0 as a fallback
            return 0

        if not qualified_children:
            # Ensure untried_actions is not empty before choosing
            return root.untried_actions[0] if root.untried_actions else 0

        if not qualified_children:
            qualified_children = root.children
        if not qualified_children:
            return np.random.choice(root.untried_actions)
        values = np.array([c.value / c.visits for c in qualified_children])
        visits = np.array([c.visits for c in qualified_children])
        if len(values) > 0:
            values = values - np.min(values)
            value_range = np.max(values) - np.min(values)
            if value_range > 0:
                values = values / value_range
        visit_scores = np.log1p(visits) / np.log1p(np.max(visits))
        combined_scores = 0.7 * values + 0.3 * visit_scores
        if temperature <= 0:
            chosen_idx = np.argmax(combined_scores)
        else:
            exp_values = np.exp(combined_scores / temperature)
            probs = exp_values / np.sum(exp_values)
            chosen_idx = np.random.choice(len(qualified_children), p=probs)
        chosen_action = qualified_children[chosen_idx].action
        if chosen_action is None:
            # Fallback to first available action or 0
            return root.untried_actions[0] if root.untried_actions else 0

        self._last_action = chosen_action
        return chosen_action

    def train(
        self, num_episodes: int = 1000
    ) -> tuple[list[float], list[int], list[float], list[float]]:
        scores = []
        steps_per_episode = []
        epsilon_values = []
        action_times = []
        for episode in range(num_episodes):
            self._episode_count = episode
            state = self.env.reset()
            done = False
            steps = 0
            episode_action_times = []
            cumulative_reward = 0
            self.current_epsilon = max(
                self.min_epsilon, self.initial_epsilon * (self.epsilon_decay**episode)
            )
            while not done and steps < self.max_depth:
                start_time = time.time()
                action = self.choose_action(state)
                action_time = time.time() - start_time
                episode_action_times.append(action_time)
                state, reward, done, _ = self.env.step(action)
                cumulative_reward += reward
                steps += 1
            if cumulative_reward < 0:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
            scores.append(cumulative_reward)
            steps_per_episode.append(steps)
            epsilon_values.append(self.current_epsilon)
            action_times.append(np.mean(episode_action_times))
            if (episode + 1) % 1 == 0:
                avg_score = np.mean(scores[-100:] if len(scores) > 100 else scores)
                avg_action_time = np.mean(action_times[-100:] if len(action_times) > 100 else action_times)
                avg_epsilon = np.mean(epsilon_values[-100:])
                logging.info(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Steps: {steps}, "
                    f"Avg Action Time: {avg_action_time:.4f}s, "
                    f"Epsilon: {avg_epsilon:.4f}"
                )
        return scores, steps_per_episode, epsilon_values, action_times
