from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass