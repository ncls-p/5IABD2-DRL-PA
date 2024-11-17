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
        self, env: Environment, num_simulations: int = 1000, max_depth: int = 50,
        exploration_weight: float = 8.0, simulation_temp: float = 4.0
    ):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.player = 1
        self.exploration_weight = exploration_weight
        self.simulation_temp = simulation_temp
        # Cache for available actions to reduce environment calls
        self._available_actions = None
        self._last_state = None
        # Value statistics with double normalization
        self._value_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0.0,
            'count': 0,
            'mean': 0.0,
            'var': 0.0,
            'recent_values': [],  # Track recent values for adaptive exploration
            'best_value': float('-inf')
        }
        # Simulation cache with decay
        self._sim_cache = {}
        self._cache_hits = 0
        self._total_sims = 0
        self._episode_count = 0
        self._min_exploration = 0.5  # Increased minimum exploration
        self._recent_window = 100  # Window for tracking recent performance
        self._last_action = None
        self._consecutive_failures = 0

    def copy_environment(self, env: Environment) -> Environment:
        return deepcopy(env)

    def get_available_actions(self, state: Any) -> np.ndarray:
        """Get available actions with caching for performance."""
        if state != self._last_state:
            self._available_actions = self.env.available_actions()
            self._last_state = state
        return self._available_actions

    def update_value_stats(self, value: float) -> None:
        """Update running statistics with online variance."""
        self._value_stats['count'] += 1
        count = self._value_stats['count']
        
        # Update min/max
        self._value_stats['min'] = min(self._value_stats['min'], value)
        self._value_stats['max'] = max(self._value_stats['max'], value)
        self._value_stats['sum'] += value
        
        # Welford's online algorithm for mean and variance
        delta = value - self._value_stats['mean']
        self._value_stats['mean'] += delta / count
        delta2 = value - self._value_stats['mean']
        self._value_stats['var'] += delta * delta2

    def normalize_value(self, value: float) -> float:
        """Double normalization using mean, variance, and bounds."""
        if self._value_stats['count'] < 2:
            return value
            
        # Z-score normalization
        std = np.sqrt(self._value_stats['var'] / (self._value_stats['count'] - 1))
        if std == 0:
            z_score = 0
        else:
            z_score = (value - self._value_stats['mean']) / std
            
        # Bound normalization
        v_range = self._value_stats['max'] - self._value_stats['min']
        if v_range == 0:
            return z_score
            
        v_norm = (value - self._value_stats['min']) / v_range
        
        # Combine both normalizations
        return 0.5 * (z_score + v_norm)

    def simulate(self, env: Environment, max_steps: int = 50) -> float:
        """Optimistic simulation with goal-directed exploration."""
        state_hash = hash(str(env.state))
        self._total_sims += 1
        
        if state_hash in self._sim_cache:
            self._cache_hits += 1
            cached_value = self._sim_cache[state_hash]
            # Add optimistic noise
            noise_scale = 1.0 / (1 + self._episode_count / 500.0)
            noise = np.random.uniform(0, noise_scale)  # Only positive noise
            return cached_value + noise
            
        steps = 0
        total_reward = 0
        discount = 0.99  # Higher discount to encourage long-term planning
        
        # Get initial state info
        if hasattr(env, 'state') and hasattr(env, 'target_position'):
            start_pos = env.state
            goal_pos = env.target_position
            distance_to_goal = abs(goal_pos - start_pos)
        
        while not env.is_game_over() and steps < max_steps:
            available_actions = env.available_actions()
            if len(available_actions) == 0:
                break

            # Goal-directed exploration
            if hasattr(env, 'state') and hasattr(env, 'target_position'):
                current_pos = env.state
                # Bias towards moving towards goal
                if current_pos < env.target_position:
                    right_bonus = 0.5
                    left_bonus = -0.5
                else:
                    right_bonus = -0.5
                    left_bonus = 0.5
            else:
                right_bonus = left_bonus = 0
                
            # Progressive exploration rate
            base_rate = max(
                self._min_exploration,
                2.0 / (1.0 + steps / 10.0)  # Slower decay with steps
            )
            
            if np.random.random() < base_rate:
                action = np.random.choice(available_actions)
            else:
                # Value-based selection with directional bias
                values = np.ones(len(available_actions))
                for i, a in enumerate(available_actions):
                    if a == 0:  # left
                        values[i] += left_bonus
                    else:  # right
                        values[i] += right_bonus
                
                temp = max(0.5, self.simulation_temp / (1 + steps/10))
                probs = np.exp(values / temp)
                probs = probs / np.sum(probs)
                action = np.random.choice(available_actions, p=probs)
            
            _, reward, done, _ = env.step(action)
            
            # Shaped reward to encourage progress
            if hasattr(env, 'state') and hasattr(env, 'target_position'):
                current_distance = abs(env.target_position - env.state)
                progress_reward = (distance_to_goal - current_distance) * 0.1
                shaped_reward = reward + progress_reward
            else:
                shaped_reward = reward
                
            total_reward += (discount ** steps) * shaped_reward
            steps += 1
            if done:
                # Extra reward for reaching goal
                if reward > 0:
                    total_reward += 2.0
                break
        
        # Update best value seen
        if total_reward > self._value_stats['best_value']:
            self._value_stats['best_value'] = total_reward
        
        # Cache result
        if steps > 0:
            self._sim_cache[state_hash] = total_reward
            if len(self._sim_cache) > 10000:
                self._sim_cache.clear()
        
        return total_reward

    def choose_action(self, state: Any, temperature: float = 1.0) -> int:
        """Action selection with goal-directed exploration."""
        root = MCTSNode(state)
        available_actions = self.get_available_actions(state)
        root.untried_actions = list(available_actions)
        
        if len(root.untried_actions) == 1:
            return root.untried_actions[0]
            
        # Get current state info
        if hasattr(self.env, 'state') and hasattr(self.env, 'target_position'):
            current_pos = self.env.state
            goal_pos = self.env.target_position
            
            # Increase exploration after failures
            if self._consecutive_failures > 0:
                self.exploration_weight *= (1 + 0.1 * self._consecutive_failures)
                temperature *= (1 + 0.2 * self._consecutive_failures)
        
        for _ in range(self.num_simulations):
            node = root
            sim_env = self.copy_environment(self.env)
            path = [node]
            
            # Selection
            depth = 0
            while (
                not sim_env.is_game_over() 
                and node.is_fully_expanded() 
                and depth < self.max_depth
            ):
                total_visits = max(1, sum(c.visits for c in node.children))
                log_total = np.log(total_visits)
                
                # UCB with goal-directed bias
                values = []
                for child in node.children:
                    # Base UCB value
                    value = child.value / max(1, child.visits)
                    explore_term = self.exploration_weight * np.sqrt(log_total / child.visits)
                    
                    # Add directional bias
                    if hasattr(sim_env, 'state') and hasattr(sim_env, 'target_position'):
                        if child.action == 1 and sim_env.state < sim_env.target_position:
                            value += 0.5  # Bonus for moving right towards goal
                        elif child.action == 0 and sim_env.state > sim_env.target_position:
                            value += 0.5  # Bonus for moving left towards goal
                    
                    values.append(value + explore_term)
                
                node = node.children[np.argmax(values)]
                path.append(node)
                
                if node.action is not None:
                    sim_env.step(node.action)
                depth += 1

            # Expansion
            if not sim_env.is_game_over() and not node.is_fully_expanded():
                action = node.untried_actions.pop()
                next_state, _, done, _ = sim_env.step(action)
                node = node.add_child(next_state, action)
                node.untried_actions = list(sim_env.available_actions())
                path.append(node)

            # Simulation and backpropagation
            value = self.simulate(sim_env)
            self.update_value_stats(value)
            
            # Optimistic value updates
            decay = 0.95
            for i, node in enumerate(reversed(path)):
                node.visits += 1
                node.value += value * (decay ** i)

        # Action selection with goal-directed bias
        visit_threshold = max(2, int(np.sqrt(self.num_simulations)))
        qualified_children = [
            c for c in root.children 
            if c.visits >= visit_threshold
        ]
        
        if not qualified_children:
            qualified_children = root.children
        
        if not qualified_children:
            # If no children, choose action moving towards goal
            if hasattr(self.env, 'state') and hasattr(self.env, 'target_position'):
                if self.env.state < self.env.target_position:
                    preferred_action = 1  # Move right
                else:
                    preferred_action = 0  # Move left
                if preferred_action in root.untried_actions:
                    return preferred_action
            return np.random.choice(root.untried_actions)

        # Combine value and visit information
        values = np.array([c.value / c.visits for c in qualified_children])
        visits = np.array([c.visits for c in qualified_children])
        
        # Normalize values
        if len(values) > 0:
            values = values - np.min(values)
            value_range = np.max(values) - np.min(values)
            if value_range > 0:
                values = values / value_range
        
        # Add directional bias
        if hasattr(self.env, 'state') and hasattr(self.env, 'target_position'):
            for i, child in enumerate(qualified_children):
                if child.action == 1 and self.env.state < self.env.target_position:
                    values[i] += 0.3
                elif child.action == 0 and self.env.state > self.env.target_position:
                    values[i] += 0.3
        
        # Combine with visit counts
        visit_scores = np.log1p(visits) / np.log1p(np.max(visits))
        combined_scores = 0.7 * values + 0.3 * visit_scores
        
        if temperature <= 0:
            chosen_idx = np.argmax(combined_scores)
        else:
            # Softmax selection
            exp_values = np.exp(combined_scores / temperature)
            probs = exp_values / np.sum(exp_values)
            chosen_idx = np.random.choice(len(qualified_children), p=probs)
        
        chosen_action = qualified_children[chosen_idx].action
        self._last_action = chosen_action
        return chosen_action

    def train(self, num_episodes: int = 1000) -> List[float]:
        scores = []
        steps_per_episode = []
        action_times = []
        
        for episode in range(num_episodes):
            self._episode_count = episode
            state = self.env.reset()
            done = False
            steps = 0
            episode_action_times = []
            cumulative_reward = 0
            
            while not done and steps < self.max_depth:
                start_time = time.time()
                action = self.choose_action(state)
                action_time = time.time() - start_time
                episode_action_times.append(action_time)
                
                state, reward, done, _ = self.env.step(action)
                cumulative_reward += reward
                steps += 1
            
            # Update consecutive failures
            if cumulative_reward < 0:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0
            
            scores.append(cumulative_reward)
            steps_per_episode.append(steps)
            action_times.append(np.mean(episode_action_times))
            
            # Log progress
            if (episode + 1) % 1 == 0:
                avg_score = np.mean(scores[-100:] if len(scores) > 100 else scores)
                avg_action_time = np.mean(action_times[-100:] if len(action_times) > 100 else action_times)
                logging.info(
                    f"Episode {episode + 1}/{num_episodes}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Steps: {steps}, "
                    f"Avg Action Time: {avg_action_time:.4f}s"
                )
        
        return scores, steps_per_episode, action_times
