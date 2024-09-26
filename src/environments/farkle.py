from typing import Tuple

import numpy as np

from src.environments import Environment


class Farkle(Environment):
    def __init__(self):
        """
        Initialize the Farkle environment.
        """
        self.dice = np.zeros(6, dtype=int)
        self.current_score = 0
        self.total_score = 0
        self.done = False
        self.score_value = 0
        self.turn_score = 0
        self.hot_dice = False

    def render(self) -> None:
        """
        Render the current state of the environment.
        """
        self.display()

    def num_states(self) -> int:
        """
        Get the number of states in the environment.

        Returns:
            int: The number of states (6^6 possible dice combinations).
        """
        return 6**6

    def num_actions(self) -> int:
        """
        Get the number of possible actions in the environment.

        Returns:
            int: The number of actions (64 possible combinations of keeping dice).
        """
        return 64  # 2^6 possible combinations of keeping dice

    def num_rewards(self) -> int:
        """
        Get the number of possible rewards in the environment.

        Returns:
            int: The number of rewards.
        """
        return 3  # Farkle (-1), Continue (0), Score (1)

    def reward(self, i: int) -> float:
        """
        Get the reward value for a given reward index.

        Args:
            i (int): The reward index.

        Returns:
            float: The reward value.
        """
        rewards = [-1.0, 0.0, 1.0]
        return rewards[i]

    def state_id(self) -> int:
        """
        Get the current state ID.

        Returns:
            int: The current state ID.
        """
        return self._dice_to_int(self.dice)

    def reset(self) -> int:
        """
        Reset the environment to its initial state.

        Returns:
            int: The initial state ID.
        """
        self.dice = np.random.randint(1, 7, size=6)
        self.current_score = 0
        self.total_score = 0
        self.done = False
        self.score_value = 0
        self.turn_score = 0
        self.hot_dice = False
        return self.state_id()

    def display(self) -> None:
        """
        Display the current state of the environment.
        """
        print(f"Dice: {self.dice}")
        print(f"Current turn score: {self.turn_score}")
        print(f"Total score: {self.total_score}")

    def is_forbidden(self, action: int) -> int:
        """
        Check if an action is forbidden in the current state.

        Args:
            action (int): The action to check.

        Returns:
            int: 0 if the action is allowed, 1 if the action is forbidden.
        """
        kept_dice = self._action_to_kept_dice(action)
        return int(not self._is_valid_keep(kept_dice))

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.done

    def available_actions(self) -> np.ndarray:
        """
        Get the available actions in the current state.

        Returns:
            np.ndarray: An array of available actions (1-63 for keeping dice, 0 for banking).
        """
        actions = [0]  # Always include the option to bank points
        for a in range(1, 64):
            if not self.is_forbidden(a):
                actions.append(a)
        return np.array(actions)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[int, float, bool, dict]: The next state ID, reward, done flag, and additional info.
        """
        if self.done:
            return self.state_id(), 0, True, {}

        if action == 0:  # Bank points
            self.total_score += self.turn_score
            self.current_score = self.total_score
            self.turn_score = 0
            self.dice = np.random.randint(1, 7, size=6)
            self.hot_dice = False
            return self.state_id(), self.total_score / 10000, False, {}

        kept_dice = self._action_to_kept_dice(action)
        if not self._is_valid_keep(kept_dice):
            self.done = True
            return self.state_id(), -1, True, {}

        score = self._calculate_score(kept_dice)
        self.turn_score += score

        remaining_dice = 6 - np.sum(kept_dice)
        if remaining_dice == 0:
            self.hot_dice = True
            remaining_dice = 6

        self.dice = np.random.randint(1, 7, size=remaining_dice)

        if self._is_farkle():
            self.done = True
            self.turn_score = 0
            return self.state_id(), -1, True, {}

        reward = score / 1000  # Normalize the reward
        self.score_value += reward

        return self.state_id(), reward, False, {}

    def score(self) -> float:
        """
        Get the current score value.

        Returns:
            float: The current score value.
        """
        return self.score_value

    @staticmethod
    def from_random_state() -> "Farkle":
        """
        Create a Farkle environment with a random state.

        Returns:
            Farkle: A new Farkle environment with a random state.
        """
        env = Farkle()
        env.reset()
        return env

    def _dice_to_int(self, dice: np.ndarray) -> int:
        """
        Convert a dice state to an integer representation.

        Args:
            dice (np.ndarray): The dice state.

        Returns:
            int: The integer representation of the dice state.
        """
        return sum((d - 1) * (6**i) for i, d in enumerate(dice))

    def _action_to_kept_dice(self, action: int) -> np.ndarray:
        """
        Convert an action to a kept dice array.

        Args:
            action (int): The action (1-63, representing dice to keep).

        Returns:
            np.ndarray: The kept dice array.
        """
        return np.array([(action & (1 << i)) > 0 for i in range(6)])

    def _is_valid_keep(self, kept_dice: np.ndarray) -> bool:
        """
        Check if the kept dice form a valid scoring combination.

        Args:
            kept_dice (np.ndarray): The kept dice array.

        Returns:
            bool: True if the kept dice are valid, False otherwise.
        """
        if np.sum(kept_dice) == 0:
            return False
        score = self._calculate_score(kept_dice)
        return score > 0

    def _calculate_score(self, kept_dice: np.ndarray) -> int:
        """
        Calculate the score for the kept dice.

        Args:
            kept_dice (np.ndarray): The kept dice array.

        Returns:
            int: The calculated score.
        """
        score = 0
        if len(kept_dice) != len(self.dice):
            kept_dice = kept_dice[: len(self.dice)]
        dice_values = self.dice[kept_dice.astype(bool)]
        if len(dice_values) == 0:
            return 0
        dice_counts = np.bincount(dice_values, minlength=7)[1:]

        # Three or more of a kind
        for i, count in enumerate(dice_counts, 1):
            if count >= 3:
                if i == 1:
                    score += 1000 * (2 ** (count - 3))
                else:
                    score += i * 100 * (2 ** (count - 3))

        # Individual 1s and 5s
        score += (dice_counts[0] % 3) * 100
        score += (dice_counts[4] % 3) * 50

        # Check for special combinations
        if len(dice_values) == 6:
            if np.array_equal(dice_counts, [1, 1, 1, 1, 1, 1]):  # Straight
                score = max(score, 1500)
            elif np.sum(dice_counts == 2) == 3:  # Three pairs
                score = max(score, 1500)

        return score

    def _is_farkle(self) -> bool:
        """
        Check if the roll is a "farkle" (no scoring dice).

        Returns:
            bool: True if the roll is a farkle, False otherwise.
        """
        return self._calculate_score(np.ones(len(self.dice), dtype=int)) == 0

    def _has_scoring_dice(self) -> bool:
        """
        Check if there are any scoring dice in the current roll.

        Returns:
            bool: True if there are scoring dice, False otherwise.
        """
        return self._calculate_score(np.ones(6, dtype=int)) > 0

    def state_vector(self) -> np.ndarray:
        """
        Get the current state of the game as a vector encoding.

        Returns:
            np.ndarray: The vector encoding of the current state.
        """
        return np.concatenate([self.dice, [self.current_score, self.total_score]])

    def action_vector(self, action: int) -> np.ndarray:
        """
        Get the vector encoding of the given action.

        Args:
            action (int): The action to encode.

        Returns:
            np.ndarray: The vector encoding of the action.
        """
        return self._action_to_kept_dice(action)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        """
        Get the transition probability for a given state, action, next state, and reward index.

        Args:
            s (int): The current state.
            a (int): The action taken.
            s_p (int): The next state.
            r_index (int): The reward index.

        Returns:
            float: The transition probability.
        """
        # In Farkle, transitions are stochastic due to dice rolls
        # This is a simplified implementation and may need refinement
        current_dice = self._int_to_dice(s)
        next_dice = self._int_to_dice(s_p)
        kept_dice = self._action_to_kept_dice(a)

        # Check if the action is valid for the current state
        if not self._is_valid_keep(kept_dice):
            return 1.0 if s == s_p and r_index == 0 else 0.0

        # Calculate the number of dice to be rerolled
        reroll_count = 6 - np.sum(kept_dice)
        if reroll_count == 0:
            reroll_count = 6

        # Check if the next state is possible given the action
        if not np.all(next_dice[:reroll_count] == current_dice[kept_dice == 0]):
            return 0.0

        # Calculate the probability of rolling the specific dice values
        prob = (1 / 6) ** reroll_count

        # Check if the transition results in scoring
        score = self._calculate_score(kept_dice)
        if score > 0:
            return prob if r_index == 2 else 0.0
        else:
            return prob if r_index == 0 else 0.0

    def _int_to_dice(self, state: int) -> np.ndarray:
        """
        Convert an integer state representation to a dice state.

        Args:
            state (int): The integer state representation.

        Returns:
            np.ndarray: The dice state.
        """
        return np.array([(state // (6**i)) % 6 + 1 for i in range(6)])
