import numpy as np


class ActivationFunc:
    """
    Class containing various activation functions (with their derivatives) for use with Neural Networks.
    Contained methods (method call):
        - Logistic Sigmoid (logistic_sigmoid)
        - ReLU (relu)
    """
    @staticmethod
    def logistic_sigmoid(inputs: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        Logistic Sigmoid Function.

        Parameters
        ----------
        inputs : np.ndarray
           Numerical input values, typically incoming values to a Neural Network layer.
        derivative : bool
            Whether or not to use the derivative of the function.

        Returns
        -------
        np.ndarry
            The resultant numpy array following function application.
        """
        if derivative:
            return inputs * (1 - inputs)
        return 1 / (1 + np.exp(-inputs))

    @staticmethod
    def relu(inputs: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        Rectified Linear Unit (ReLU) function.

        Parameters
        ----------
        inputs : np.ndarray
            Numerical input values, typically incoming values to a Neural Network layer.
        derivative : bool
            Whether or not to use the derivative of the function.

        Returns
        -------
        np.ndarry
            The resultant numpy array following function application.
        """
        if derivative:
            return 1 * (inputs > 0)
        return inputs * (inputs > 0)
