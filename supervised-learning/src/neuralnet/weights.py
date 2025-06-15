import numpy as np

# TODO - add return values
class WeightInit:
    """
    Class containing weight initialisation methods for use with neural networks.
    Contained methods (method call):
        - Random Initialisation (random)
        - Xavier Initialisation (xavier)
        - Kaiming He Initialisation (kaiming_he)
    """
    @staticmethod
    def random(num_input: int, num_output: int):
        """
        Initialise layer weights to be to be random numbers between zero and one, normalised such that each row
        sums to 1.
        [add reference to lab sheet code]

        Parameters
        ----------
        num_input : int
            Number of neurons in current layer of neural network.
        num_output : int
            Number of neurons in next layer of neural network.

        Returns
        -------
        matrix
            n by m matrix where n is `num_output` and m is num `num_input`
        """
        weights = np.random.uniform(0, 1, (num_output, num_input))
        normalised_weights = np.divide(weights, np.tile(np.sum(weights, 1)[:, None], num_input))
        return normalised_weights

    @staticmethod
    def xavier(num_input: int, num_output: int):
        """
        Initialise layer weights using the Xavier initialisation function.

        Parameters
        ----------
        num_input : int
            Number of neurons in current layer of neural network.
        num_output : int
            Number of neurons in next layer of neural network.

        Returns
        -------
        matrix
            n by m matrix where n is `num_output` and m is num `num_input`
        """
        return np.random.randn(num_output, num_input) * np.sqrt(1 / num_input)

    @staticmethod
    def kaiming_he(num_input: int, num_output: int):
        """
        Initialise layer weights using the Kaiming He initialisation function.

        Parameters
        ----------
        num_input : int
            Number of neurons in current layer of neural network.
        num_output : int
            Number of neurons in next layer of neural network.

        Returns
        -------
        matrix
            n by m matrix where n is `num_output` and m is num `num_input`
        """
        return np.random.randn(num_output, num_input) * np.sqrt(2 / num_input)
