from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from environment.actions import Action

import numpy as np


class Policy(ABC):
    """
    Abstract base class representing a Policy for use with
    Reinforcement Learning Tasks.
    """

    @abstractmethod
    def execute(self, state_int: int, n_actions: int, *args: Any) -> int:
        """
        Abstract method for executing a policy.
        Takes an Agents current state, list of possible actions and
        any additional parameters and returns the Action to be taken by
        the Agent.

        Parameters
        ----------
        state_int : int
            An integer representation of the agents state.
            Determined as follows:
              2D Environment -> | 1 | 2 | 3 |
                                | 4 | 5 | 6 |
                                | 7 | 8 | 9 |
              A states integer value is taken to be:
                    column_index + (row_index * row_len)
            A similar principle applies for higher dimensions, such the each
            environment cell has a unique integer.
        n_actions : int
            The number of possible actions
        args : Any
            Any additional positional arguments required by the Policy.

        Returns
        -------
         Returns
        -------
        int
            The ID (index in List) of the selected action.
        """
        pass


class EpsilonGreedy(Policy):
    def __init__(self, epsilon: float) -> None:
        """
        Initialise an instance of the ε-greedy algorithm using the
        specified parameters.
        This Policy allows for simple configuration of whether a
        Reinforcement Learning Agent should favour exploration (random
        movement to 'explore' the Environment) or exploitation (targeted movement,
        selecting the Action which seems to be the best).
        It can be summarised as follows:

            action_to_be_taken =  | 'best' action       with probability: 1 - ε
                                  |  random action      with probability: ε

        Parameters
        ----------
        epsilon : float
            The value of ε for use with the Policy.
            Higher values will cause the agent to favour random actions.
            Lower values will cause the agent to favour the 'best' action.

        Returns
        ----------
            None.
        """
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        """
        Get the value of epsilon (ε).

        Returns
        -------
        float
            The stored value for epsilon (ε).
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """
        Set a new value for epsilon (ε).

        Parameters
        ----------
        value : float
            The new value for epsilon (ε).
            Should be in the interval [0,1].

        Returns
        -------
            None.

        Raises
        -------
        ValueError
            If given value is not in the interval [0,1].
        """
        if 0 <= value <= 1:
            self._epsilon = value
        else:
            raise ValueError("Value for epsilon must be between 0 and 1, inclusive.")

    def execute(self, state_int: int, n_actions: int, q_values: np.ndarray) -> int:
        """
        Execute the ε-greedy policy to determine the action the Agent will take.

        Parameters
        ----------
        state_int : int
            An integer representation of the agents state.
            Determined as follows:
              2D Environment -> | 1 | 2 | 3 |
                                | 4 | 5 | 6 |
                                | 7 | 8 | 9 |
              A states integer value is taken to be:
                    column_index + (row_index * row_len)
            A similar principle applies for higher dimensions, such the each
            environment cell has a unique integer.
        n_actions : int
            The number of possible actions
        q_values : np.ndarray
            An N x M numpy array containing the Q-values for State, Action pairs.
            Where N is the number of States and M is the number of Actions.

        Returns
        -------
        int
            The ID (index in List) of the selected action.
        """
        if np.random.rand() > self._epsilon:
            action_id = np.argmax(q_values[state_int, :])
        else:
            action_id = np.random.randint(0, n_actions)

        return action_id
