from typing import List, Tuple, Type

from environment.actions import Action
import numpy as np


class GridWorld:
    def __init__(self, grid: np.ndarray, permitted_actions: List[Type[Action]], out_of_bounds_penalty: float = -0.1, movement_penalty: float = 0.0) -> None:
        """
        Initialised a 2-Dimensional GridWorld to serve as an Environment for
        Reinforcement Learning tasks. Uses numpy arrays as a base.

        Parameters
        ----------
        grid : numpy.ndarray
            A numpy array underlying the GridWorld. Each element of the grid should
            contain a float representing the reward of the state.
        permitted_actions : List[Action]
            The actions that can be performed by an agent in the Environment.
        out_of_bounds_penalty : float
            The reward to be received if an Agent move outside of the grid range.s
        """
        self._grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
        self.permitted_actions = permitted_actions
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.movement_penalty = movement_penalty

    def get_reward(self, state: Tuple[int, int]) -> float:
        """
        Get the reward value associated with a given state.

        Parameters
        ----------
        state : Tuple[int, int]
            The state for which the reward will be obtained.

        Returns
        -------
        float
            The reward value.
        """
        return self._grid[state[0], state[1]]
