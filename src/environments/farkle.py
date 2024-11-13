from typing import List, Tuple

import numpy as np

from src.environments import Environment


class Farkle(Environment):
    def __init__(self, target_score=10000):
        """
        Initialize the Farkle environment.

        Args:
            target_score (int): The score required to win the game.
        """
        self.dice = np.zeros(6, dtype=int)
        self.current_score = 0
        self.total_score = 0
        self.done = False
        self.score_value = 0
        self.turn_score = 0
        self.hot_dice = False
        self.target_score = target_score
        self.current_player = 1
        self.scores = [0, 0]  # Scores for player 1 and player 2
        self.final_round = False
        self.final_round_starter = None

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
        return 3  # Lose (-1), No score (0), Win (1)

    def reward(self, i: int) -> float:
        """
        Get the reward value for a given reward index.

        Args:
            i (int): The reward index.

        Returns:
            float: The reward value.
        """
        rewards = [-1.0, 0, 1.0]
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
        self.scores = [0, 0]
        self.done = False
        self.score_value = 0
        self.turn_score = 0
        self.hot_dice = False
        self.current_player = 1
        self.final_round = False
        self.final_round_starter = None
        return self.state_id()

    def display(self) -> None:
        """
        Display the current state of the environment.
        """
        print(f"Dice: {self.dice}")
        print(f"Current turn score: {self.turn_score}")
        print(f"Player 1 score: {self.scores[0]}")
        print(f"Player 2 score: {self.scores[1]}")

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
            np.ndarray: An array of available actions (1 to max_action - 1 for keeping dice, 0 for banking).
        """
        actions = [0]
        num_dice = len(self.dice)
        max_action = 2**num_dice
        for a in range(1, max_action):
            if not self.is_forbidden(a):
                actions.append(a)
        return np.array(actions, dtype=int)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): The action to take.

        Returns:
            Tuple[int, float, bool, dict]: The next state ID, reward, done flag, and additional info.
        """
        if self.done:
            reward = (
                1.0
                if self.scores[self.current_player - 1] >= self.target_score
                else -1.0
            )
            return (
                self.state_id(),
                reward,
                True,
                {"current_player": self.current_player},
            )

        reward = 0.0  # Default reward for non-terminal steps

        if action == 0:  # Bank points
            self.scores[self.current_player - 1] += self.turn_score
            self.turn_score = 0
            self.dice = np.random.randint(1, 7, size=6)
            self.hot_dice = False

            if (
                self.scores[self.current_player - 1] >= self.target_score
                and not self.final_round
            ):
                self.final_round = True
                self.final_round_starter = self.current_player

            self.current_player = 3 - self.current_player

            if self.final_round and self.current_player == self.final_round_starter:
                self.done = True
                winner = 1 if self.scores[0] >= self.scores[1] else 2
                reward = 1.0 if winner == self.current_player else -1.0
                return (
                    self.state_id(),
                    reward,
                    True,
                    {"current_player": self.current_player},
                )

            return (
                self.state_id(),
                reward,
                False,
                {"current_player": self.current_player},
            )

        kept_dice = self._action_to_kept_dice(action)

        if not self._is_valid_keep(kept_dice):
            self.turn_score = 0
            self.current_player = 3 - self.current_player
            self.dice = np.random.randint(1, 7, size=6)
            return (
                self.state_id(),
                reward,
                False,
                {"current_player": self.current_player},
            )

        score = self._calculate_score(kept_dice)
        self.turn_score += score

        remaining_dice = len(self.dice) - np.sum(kept_dice)
        if remaining_dice == 0:
            self.hot_dice = True
            remaining_dice = 6

        # Roll the remaining dice
        self.dice = np.random.randint(1, 7, size=remaining_dice)

        return self.state_id(), reward, False, {"current_player": self.current_player}

    def score(self) -> List[int]:
        """
        Get the current score values for both players.

        Returns:
            List[int]: The current score values for player 1 and player 2.
        """
        return self.scores

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
            action (int): The action (1 to max_action - 1).

        Returns:
            np.ndarray: The kept dice array.
        """
        num_dice = len(self.dice)
        return np.array([(action & (1 << i)) > 0 for i in range(num_dice)], dtype=bool)

    def _is_valid_keep(self, kept_dice: np.ndarray) -> bool:
        """
        Check if the kept dice constitute a valid scoring combination.

        Args:
            kept_dice (np.ndarray): Boolean array indicating which dice are kept.

        Returns:
            bool: True if the kept dice are valid, False otherwise.
        """
        if np.sum(kept_dice) == 0:
            return False
        dice_values = self.dice[kept_dice]
        if len(dice_values) == 0:
            return False
        if len(dice_values) == len(self.dice):
            # Only allow keeping all dice if it's a straight or three pairs
            dice_counts = np.bincount(dice_values, minlength=7)[1:]
            if not (
                np.array_equal(dice_counts, [1, 1, 1, 1, 1, 1])
                or np.sum(dice_counts == 2) == 3
            ):
                return False
        score = self._calculate_score(kept_dice)
        return score > 0

    def _calculate_score(self, kept_dice: np.ndarray) -> int:
        score = 0
        dice_values = self.dice[kept_dice.astype(bool)]
        if len(dice_values) == 0:
            return 0
        dice_counts = np.bincount(dice_values, minlength=7)[1:]

        # Check for special combinations first
        if len(dice_values) == 6:
            if np.array_equal(dice_counts, [1, 1, 1, 1, 1, 1]):  # Straight
                return 1500
            elif np.sum(dice_counts == 2) == 3:  # Three pairs
                return 1500

        # Score three or more of a kind
        for i, count in enumerate(dice_counts, 1):
            if count >= 3:
                base_score = 1000 if i == 1 else i * 100
                score += base_score * (2 ** (count - 3))
                dice_counts[i - 1] = 0  # Remove these dice from further scoring

        # Score remaining 1s and 5s
        score += dice_counts[0] * 100  # Remaining 1s
        score += dice_counts[4] * 50  # Remaining 5s

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
        Get the simplified state vector for MDP algorithms.

        Returns:
            np.ndarray: The state vector containing essential information.
        """
        # Counts of each dice value (1-6)
        dice_counts = np.bincount(self.dice, minlength=7)[1:]
        # State vector with essential information
        state = np.concatenate(
            [
                dice_counts,  # Dice counts
                [self.turn_score],  # Current turn score
                [self.scores[self.current_player - 1]],  # Current player's total score
            ]
        )
        return state.astype(float)

    def action_vector(self, action: int) -> int:
        """
        Represent the action as an integer index.

        Args:
            action (int): The action index.

        Returns:
            int: The action index.
        """
        return action

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
        if not np.all(next_dice[:reroll_count] == current_dice[kept_dice is False]):
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