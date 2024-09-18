from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

class Environment(ABC):
    @abstractmethod
    def reset(self) -> Any:
        pass

    @abstractmethod
    def step(self, action: Any) -> Union[Tuple[Any, float, bool, dict], Any]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def num_states(self) -> int:
        pass

    @abstractmethod
    def num_actions(self) -> int:
        pass

    @abstractmethod
    def num_rewards(self) -> int:
        pass

    @abstractmethod
    def reward(self, i: int) -> float:
        pass

    @abstractmethod
    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        pass

    @abstractmethod
    def state_id(self) -> int:
        pass

    @abstractmethod
    def is_forbidden(self, action: int) -> int:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def available_actions(self) -> Any:
        pass

    @abstractmethod
    def score(self) -> float:
        pass

