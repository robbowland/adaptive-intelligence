from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Type
from agent.policy import Policy
from environment.environment import GridWorld

import numpy as np


class GridWorldAgent:
    def __init__(self, environment: GridWorld, initial_state: Tuple, policy: Policy,
                 q_value_init_method: str, learning_rate, discount_factor) -> None:
        """
        Initialise a Reinforcement Learning Agent using the specified parameters.
        This Agent is intended to be used with a 2D GridWorld as its environment.

        Parameters
        ----------
        environment : GridWorld
        initial_state : Tuple
            An n-tuple representing the Agents initial state in the environment.
            Note that this used numpy indexing format, so (0,0) would be top left
            on the grid.
        policy : Policy
            The Policy to be used by the Agent to determine its Actions.

        Returns
        ----------
            None.
        """
        self.environment = environment
        self.initial_state = initial_state
        self.policy = policy
        self.actions = self.environment.permitted_actions
        # Attributes related to learning
        self.q_table = self._init_q_values(q_value_init_method)  # Stores Q-values
        self.eligibility_trace_table = np.zeros((self.environment.rows * self.environment.cols, len(self.actions)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Attributes related to each run
        self.state = initial_state  # starts as the initial_state
        self.action_id = self.policy.execute(
            self._state_to_int_state(self.initial_state),
            len(self.actions),
            self.q_table
        )

    def _init_q_values(self, init_method: str) -> np.ndarray:
        """
        Initialise an N x M numpy array of Q-values according to the
        specified method. Where, N is the total number of States in the
        Environment and M is the number of possible Actions in the
        Environment.
        Intended for internal use only.

        Parameters
        ----------
        init_method : str
            A string indicating the method that should be used to initialise the Q-values.
            Should be one of:
                "zeroes" - all values initialised to 0
                "one"    - all values initialised to 1

        Returns
        -------
        numpy.ndarray
            The initialised numpy array of Q-values.
        """
        if init_method == "zeroes":
            return np.zeros((self.environment.rows * self.environment.cols, len(self.actions)))
        if init_method == "ones":
            return np.ones((self.environment.rows * self.environment.cols, len(self.actions)))
        if init_method == "random":
            return np.random.rand(self.environment.rows * self.environment.cols, len(self.actions))

    # TODO - not sure how to handle rollbacks on position when it comes to updating Q-values
    def act(self, lambda_: float = None, punish: bool = False) -> bool:
        """
        Trigger Agent action, transitioning to next state and updating Q-values based
        on environmental feedback.

        Parameters
        ----------
        lambda_
        punish : bool
            Whether or not the agent is being punished (e.g. for exceeding
            allowed number of steps). With give a reward of -1 and trigger
            an end state.

        Returns
        -------
        bool
            True if an end condition reached (goal reached or punished), False otherwise.
        """
        # ==== EXECUTE POLICY
        # Calculate the integer representation of the agents current state
        state_int = self._state_to_int_state(self.state)
        # Determine the action the agent will take

        # ==== PERFORM ACTION
        # Move the agent to the new state according to the selected action
        new_state = self.actions[self.action_id].perform(self.state)

        # ==== GET FEEDBACK & LEARN
        # Punish agent e.g. if number of steps has been exceeded without reaching end state
        if punish:
            reward = -1 + self.environment.movement_penalty
            self._learn(new_state, reward, end=True)
            return True

        # Determine whether or not the move was legal (i.e. did it go off the edge)
        if (0 <= new_state[0] < self.environment.rows) and (0 <= new_state[1] < self.environment.cols):
            # Get reward from grid
            reward = self.environment.get_reward(new_state)
            # Q-VALUE UPDATING
            # Check if end state
            if reward == 1.0:
                self._learn(new_state, reward + self.environment.movement_penalty, lambda_=lambda_, end=True)
                return True
            # Check if reward is negative i.e. an obstacle has been hit
            elif reward < 0:
                self._learn(self.state, reward + self.environment.movement_penalty, lambda_=lambda_)
                return False
            # Standard update procedure
            else:
                # reward += -0.1
                self._learn(new_state, reward + self.environment.movement_penalty, lambda_=lambda_)
                self.state = new_state
                return False
        # Agent has gone out of bounds
        else:
            reward = self.environment.out_of_bounds_penalty
            self._learn(self.state, reward + self.environment.movement_penalty, lambda_=lambda_)
            return False

    def _learn(self, new_state, reward, lambda_: float = None, end: bool = False) -> None:
        """
        Perform actions an update internal data.
        Intended for internal use only.
        """
        # ints for indexing the Q-value store
        current_state_int = self._state_to_int_state(self.state)
        new_state_int = self._state_to_int_state(new_state)
        # Q-values for current state and best expected in next state
        current_state_q = self.q_table[current_state_int, self.action_id]

        if end:  # If the agent has reach a terminal state.
            if lambda_:
                self.eligibility_trace_table[current_state_int, self.action_id] += 1
                self.q_table += self.learning_rate * (reward - current_state_q) * self.eligibility_trace_table
            else:
                self.q_table[current_state_int, self.action_id] += self.learning_rate * (reward - current_state_q)
            return

        # Choose A' (next_action) from S' (new state) based on policy -> Q(S', A')
        next_action_id = self.policy.execute(new_state_int, len(self.actions), self.q_table)
        new_state_q = self.q_table[new_state_int, next_action_id]
        delta = (reward + (self.discount_factor * new_state_q) - current_state_q)
        # Update the Q-value for the given state and action

        # Using SARSA LAMBDA
        if lambda_:
            self.eligibility_trace_table[current_state_int, self.action_id] += 1

            self.q_table += self.learning_rate * delta * self.eligibility_trace_table
            self.eligibility_trace_table *= self.discount_factor * lambda_ * self.eligibility_trace_table

        # Using SARSA
        else:
            self.q_table[current_state_int, self.action_id] += self.learning_rate * delta

        self.action_id = next_action_id

    def _state_to_int_state(self, state: Tuple[int, int]) -> int:
        """
        Utility function to convert a 2D state tuple into a single integer for
        use with indexing the Q-value table, and the Eligibility Trace table.
        Intended for internal use only.

        Parameters
        ----------
        state : Tuple[int, int]
            The 2D state Tuple to be converted.

        Returns
        -------
        int
            The integer representation of the state.
        """
        return state[1] + (state[0] * self.environment.cols)

    def reset(self) -> None:
        """
        Resets Agents state to its initial state, without wiping q-values etc.

        Returns
        -------
            None.
        """
        self.state = self.initial_state
        self.eligibility_trace_table = np.zeros((self.environment.rows * self.environment.cols, len(self.actions)))
        self.action_id = self.policy.execute(
            self._state_to_int_state(self.initial_state),
            len(self.actions),
            self.q_table
        )
