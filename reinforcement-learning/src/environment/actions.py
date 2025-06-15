from abc import ABC, abstractmethod
from typing import Any, Tuple


class Action(ABC):
    """
    Abstract base class representing an action that can be taken
    by an Agent during a Reinforcement Learning task.
    """

    @staticmethod
    @abstractmethod
    def perform(state: Any) -> Any:
        """
        Perform the specified action.

        Parameters
        ----------
        state : Any
            The current State of the agent, represented in the form suitable
            for the agents environment.

        Returns
        -------
            The new State of the agent following the performed action, represented in
            the form suitable for the agents environment.
        """
        pass


class North(Action):
    @staticmethod
    def perform(state: Tuple[int, int]) -> Tuple[int, int]:
        return state[0] - 1, state[1]


class East(Action):

    @staticmethod
    def perform(state: Tuple[int, int]) -> Tuple[int, int]:
        return state[0], state[1] + 1


class South(Action):

    @staticmethod
    def perform(state: Tuple[int, int]) -> Tuple[int, int]:
        return state[0] + 1, state[1]


class West(Action):

    @staticmethod
    def perform(state: Tuple[int, int]) -> Tuple[int, int]:
        return state[0], state[1] - 1
