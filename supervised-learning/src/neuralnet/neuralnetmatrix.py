import math
import numpy as np
from typing import List, Union, Dict, TypedDict, Callable, Tuple, Any, Optional


class NNLayer(TypedDict):
    """
    Dictionary representing a single layer of a Neural Network.

    (Key : Value)
    ----------
    (dimension : int)
        Integer dimension representing the number of neurons in the layer.
    (activation_func : Union[Callable[[np.ndarray, bool], np.ndarray], None])
        Activation function for the neurons in the layer of the network. Should have an optional derivative mode
        as seen in those from the ActivationFunc class.
        Note: This is only optional for the first element (the Input Layer) any non-None value for the Input Layer will
              be ignored regardless.
    """
    dimension: int
    activation_func: Union[Callable[[np.ndarray, bool], np.ndarray], None]


# Some of the aspects of this class are not as well organised as I would have liked but running times and complications
# meant that I couldn't spend more time organising.
class NeuralNet:
    """
    Class for creating, training and testing, Neural Networks of various sizes.
    This implementation makes use of matrix operations for batches to speed up operation.

    Attributes
    ----------
    layers : List[NNLayer]
        List of NNLayer, each representing a layer of the neural network. Stores the number of Neurons in the Layer,
        as well and the activation function to be used with Neurons on in the layer.
    weight_init_method : Callable[[int, int], np.ndarray]
        The method that used to initialise the weights of the connections in the Neural Network.
    weights_biases : Dict[str, Dict[str, np.ndarray]]
        Dictionary containing numpy arrays that store the weights and biases for the connections between each layer in
        the Neural Network.
        Each weight/bias array for given connections is stored in a Dictionary, under a key determined by the two
        Network Layer indexes that make up the connections.
        e.g. For Input Layer -> Next Layer weights/biases the key would be `0_1`.
        Weights and biases can then be accessed from the corresponding Dictionary using the keys "weights" ann "biases"
        respectively.
    """

    def __init__(self, architecture: List[NNLayer], weight_init_method: Callable[[int, int], np.ndarray]) -> None:
        """
        Initialise a Neural Network in accordance with the specified parameters.
        Weights are initialised according to the specified initialisation method.
        Biases are all initialised to zero.

        Parameters
        ----------
        architecture : List[NNLayer]
            A List of NNLayers (Dicts) representing the overall structure of the Neural Network, with each element
            of the List representing a layer of neurons in the network in the order they should appear.
        weight_init_method : Callable[[int, int], np.ndarray]
            The function to be used for initialising the weights of the Neural Network.
            Example functions can be found in the WeightInit class.
        """
        self.layers = architecture
        self.weight_init_method = weight_init_method
        self.weights_biases = {
            f"{layer_num}_{layer_num + 1}": {
                "weights": self.weight_init_method(self.layers[layer_num]["dimension"],
                                                   self.layers[layer_num + 1]["dimension"]),
                "biases": np.zeros((self.layers[layer_num + 1]["dimension"],))
            }
            for layer_num, _ in enumerate(self.layers[:-1])
        }

    # This function has become bigger than I'd like, normally I'd look at extracting some of the functionality,
    # but it functions as it should and produces the necessary stats, so for the time being I'll leave it like this,
    # given time constraints.
    def train(self, training_data: np.ndarray, training_data_labels: np.ndarray, num_epochs: int,
              batch_size: int, num_validation: int, learning_rate: float, l1_reg_lambda: float,
              stats: Tuple[bool, bool, bool, Tuple[bool, float], bool] = (False, False, False, (False, 0.0), False)
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[dict]]]:
        """
        Train the Neural Network using specified parameters.
        This function uses matrix operations to improve performance.

        Parameters
        ----------
        training_data : np.ndarray
            The data to be used for training.
        training_data_labels : np.ndarray
            The corresponding labels for the training data.
        num_epochs : int
            The number of Epochs to be used during training. This is the number of times the training algorithm
            will work through the entire training data set.
        batch_size : int
            The size of an individual batch to be used with mini-batch gradient descent.
            Note: The remaining data used for training (training data - num validation) should be divisible by this.
        num_validation : int
            The number of samples from the specified training data that will be used as the validation set for
            monitoring network training performance.
        learning_rate : float
            The learning rate of the Network.
        l1_reg_lambda : float

        stats : Tuple[bool, bool, bool, Tuple[bool, float], bool]
            A 5-Tuple used to specify which training statistics should be calculated and returned. When set to True the
            corresponding metrics are calculated and returned. Enabling these can affect network training performance.
            In Tuple index order they correspond to:
                 0 - Validation Errors
                    Whether or not to calculate and store the errors during the validation stage using the validation
                    data set. This will be the Mean Squared Error + L1 Regularisation Error (providing l1_reg_lambda
                    != 0).
                1 - Validation Accuracy
                    Whether or not to calculate and store the accuracy during the validation stage using the validation
                    date. This is the proportion of samples in the validation set that were correctly classified.
                2 - Training Errors
                    Whether or not to calculate and store the errors during the training stage using the training
                    data set. This will be the Mean Squared Error + L1 Regularisation Error (providing l1_reg_lambda
                    != 0).
                3 - (Average Weight Updates Matrix, Tau)
                    A Tuple (bool, float), the first element of which specifies whether or not the average weight
                    updates for the Neural Network layers should be calculated and stored.
                    The second element specifies the τ (Tau) value to be used during the exponential moving average
                    calculation for each of the weighted connections, as given below:

                        A_n =  |    deltaW_0                               for n = 0
                               |    A_(n-1) * (1 - τ) + (τ * deltaW_n)     for n > 0

                    Where A_n is the the Average Weight Matrix for a given layer for a given Epoch, n, (here indexed to
                    begin at 0) and deltaW_n is the Weight Update Matrix used to update the given layers Weight values
                    during the given Epoch. τ is the constant mentioned previously, usually a small value such
                    as 0.01 [1].
                    [1] M. Ellis and E. Vasilaki, Lecture notes for Adaptive Intelligence:Lecture 2, 2021.
                4 - Print to Terminal?
                    Whether or not to print the previously specified statistics to the terminal as training is
                    progressing. Only those that were previously set to true will be printed.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[dict]]]
            The specified statistics from the 4 elements of the input `stats` Tuple.
        """

        # TODO (2) - Clean up all the transposes so it's a bit clearer what's going on, or make sure they're well commented
        v_errors, v_accuracy, t_errors, w_updates, print_to_terminal = stats
        # Initialise storage for OPTIONAL STATS
        validation_errors = np.zeros((num_epochs,)) if v_errors else None
        validation_accuracy = np.zeros((num_epochs,)) if v_accuracy else None
        training_errors = np.zeros((num_epochs,)) if t_errors else None
        # weight updates for each layer at each epoch.
        average_weight_updates = [
            {key: np.zeros(data["weights"].shape) for (key, data) in self.weights_biases.items()}
            for _ in range(0, num_epochs)
        ] if w_updates else None

        # Split data set into TRAINING SET and VALIDATION SET
        num_samples = training_data.shape[0]
        num_training_samples = num_samples - num_validation
        shuffled_idxs = np.random.permutation(num_samples)
        training_set = training_data[shuffled_idxs[:num_training_samples]]
        training_set_labels = training_data_labels[shuffled_idxs[:num_training_samples]]
        validation_set = training_data[shuffled_idxs[num_training_samples:]].T
        validation_set_labels = training_data_labels[shuffled_idxs[num_training_samples:]].T

        # Determine number of batches used for mini-batch updating
        num_batches = int(math.ceil(num_training_samples / batch_size))

        # TRAIN the model.
        for epoch in range(0, num_epochs):
            shuffled_training_idxs = np.random.permutation(num_training_samples)  # Shuffle training data indexes
            for batch in range(0, num_batches):
                # -------------------------------
                # 1. PREPARING THE DATA FOR THE CURRENT BATCH
                # Create a mini-batch matrix containing all the samples in the batch
                mini_batch_matrix_idxs = shuffled_training_idxs[(batch * batch_size):(batch + 1) * batch_size]
                mini_batch_matrix = training_set[
                    mini_batch_matrix_idxs].T  # Need to transpose for Linear Alg  TODO - Comment on the expected form of the training_data
                target_outputs = training_set_labels[mini_batch_matrix_idxs].T
                # 2. FEEDFORWARD TO CALCULATE LAYER OUTPUTS
                layer_outputs = self._feedforward(mini_batch_matrix)
                # 3. BACKPROPAGATE TO CALCULATE NODE DELTA/GRADIENTS FOR WEIGHT/BIAS UPDATES
                node_deltas, gradients = self._backpropagate(layer_outputs, target_outputs)
                # 4. UPDATE NETWORK WEIGHTS & BIASES
                weight_updates, _ = self._update_weights_biases(learning_rate, node_deltas,
                                                                gradients, batch_size, l1_reg_lambda,
                                                                return_stats=True)
                # -------------------------------
                # (OPTIONAL) UPDATE STATS STORAGE
                # Update the TRAINING ERROR
                if t_errors:
                    t_mean_squared_error = 0.5 * np.sum(np.square(-(target_outputs - layer_outputs[-1])))
                    t_l1_error = self._l1_regularisation(l1_reg_lambda, derivative=False)
                    training_errors[epoch] = training_errors[epoch] + \
                                             ((t_mean_squared_error + t_l1_error) / num_training_samples)
                # Update the AVERAGE WEIGHT MATRIX
                calculate_average_weight_updates, tau = w_updates
                if calculate_average_weight_updates:
                    for key in self.weights_biases.keys():
                        if epoch == 0:
                            average_weight_updates[epoch][key] += weight_updates[key] / num_batches
                        else:
                            average_weight_updates[epoch][key] += (average_weight_updates[epoch - 1][key] * (
                                        1 - tau) + (tau * weight_updates[key])) / num_batches

            # -------------------------------
            # (OPTIONAL) PERFORM VALIDATION
            if v_errors or v_accuracy:
                v_e, v_a = self._validate(validation_set, validation_set_labels,
                                          l1_reg_lambda, errors=v_errors, accuracy=v_accuracy)
                if v_errors:
                    validation_errors[epoch] = v_e
                if v_accuracy:
                    validation_accuracy[epoch] = v_a

            # -------------------------------
            # (OPTIONAL) PRINT STATS TO CONSOLE
            if print_to_terminal:
                print("-" * 48)
                print(f"| EPOCH: {epoch + 1}")
                print("=" * 48)
                if t_errors:
                    print("| {:<26} : {:>15.5f} |".format("Training Set Error", training_errors[epoch]))
                if v_errors:
                    print("| {:<26} : {:>15.5f} |".format("Validation Set Error", validation_errors[epoch]))
                if v_accuracy:
                    print("| {:<26} : {:>14.2f}% |".format("Validation Set Accuracy", validation_accuracy[epoch] * 100))
                if w_updates[0]:
                    print("|-{:<12}Average Weight Updates{:>10}-|".format("", ""))
                    for key in self.weights_biases.keys():
                        print("| {:<26} : {:>15.5f} |".format(f"{key}", np.sum(
                            np.sum(average_weight_updates[epoch][key], axis=0))))
                print("=" * 48)
                print("\n")

        return validation_errors, validation_accuracy, training_errors, average_weight_updates

    def _validate(self, validation_data: np.ndarray, validation_data_labels: np.ndarray, l1_reg_lambda: float,
                  errors: bool = False,
                  accuracy: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """
        Perform Network validation on a validation data set in oder to monitor network error and accuracy during
        training.

        Parameters
        ----------
        validation_data : np.ndarry
            The data to be used for validation.
        validation_data_labels : np,ndarray
            The corresponding labels for the validation set.
        l1_reg_lambda : float
            The penalty factor for L1 Regularisation.
        errors : bool
            Whether or not to calculate validation errors. This will be the Mean Squared Error +
             L1 Regularisation Error (providing l1_reg_lambda != 0).
        accuracy : bool
            Whether or not to calculate validation accuracy. This is the proportion of samples in the validation
            set that were correctly classified.

        Returns
        -------
        Tuple[Optional[float], Optional[float]]
            Tuple containing the calculated metrics. The first element corresponds to the error, the second the
            accuracy. Respective element will be `None` if it has been specified not to calculate it.
        """
        layer_outputs = self._feedforward(validation_data)
        if errors:
            validation_mse = 0.5 * np.sum(np.square(-(validation_data_labels - layer_outputs[-1])))
            validation_l1e = self._l1_regularisation(l1_reg_lambda, derivative=False)
        validation_error = ((validation_mse + validation_l1e) / validation_data.shape[1]) if errors else None
        validation_accuracy = np.sum(
            np.argmax(layer_outputs[-1], axis=0) == np.argmax(validation_data_labels, axis=0)) / \
                              validation_data.shape[1] if accuracy else None

        return validation_error, validation_accuracy

    def _feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Feedforward the initial network inputs through subsequent layers and calculate the output of each Network Layer.
        This function uses matrix operations to improve performance.
        Intended for internal use only.

        Parameters
        ----------
        inputs : np.ndarray
            The initial inputs to the Neural Network i.e. those being input to the Input Layer of the Network.
            Normalisation of the data should have been performed prior to this.

        Returns
        -------
        np.ndarray[np.ndarray]
            A numpy array containing the output values for each of the Network layers.
        """
        layer_inputs = np.empty(len(self.layers), dtype=object)
        layer_inputs[0] = inputs  # Input of Input Layer == Initial Input
        for layer_num, layer in enumerate(self.layers[:-1]):  # Feed inputs forward through network layers
            layer_inputs[layer_num + 1] = self._activate(layer_inputs[layer_num],
                                                         self.weights_biases[f"{layer_num}_{layer_num + 1}"]["weights"],
                                                         self.weights_biases[f"{layer_num}_{layer_num + 1}"]["biases"],
                                                         self.layers[layer_num + 1]["activation_func"])
        # These are more accurately described as the outputs of each layer at this point
        return layer_inputs

    def _activate(self, input_values: np.ndarray, weights: np.ndarray, biases: np.ndarray,
                  activation_func: Callable[[np.ndarray, bool], np.ndarray]) -> np.ndarray:
        """
        Calculate the weighted sum of inputs for the neurons in the layer and then apply the activation function to
        calculate the outputs for the layer.
        This function uses matrix operations to improve performance.
        Intended for internal use only.

        Parameters
        ----------
        input_values : np.ndarray
            The inputs values for the Network Layer.
        weights : np.ndarray
            The relevant connection weights for the current Network layer, for calculating the weighted sum of inputs.
        biases : np.ndarray
            The relevant connection biases for the current Network layer, for calculating the weighted sum of inputs.
        activation_func : Callable[[np.ndarray, bool], np.ndarray]
            The activation function for the Neurons in the current layer.

        Returns
        -------
        np.ndarray
            The output of the neurons after the activation function has been applied to the weighted sum of the inputs.
        """
        # Calculate the weighted input to the layer
        # print(f"Weights: {weights.shape} @ inputs_values{input_values.shape}: + biases: {biases.reshape(biases.shape[0], 1).shape}")
        weighted_inputs = (weights @ input_values) + biases.reshape(biases.shape[0], 1)
        # Apply activation function and return outputs of the layer
        return activation_func(weighted_inputs, False)

    def _backpropagate(self, layer_outputs: np.ndarray,
                       target_output: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Perform backpropagation through the Neural Network in order to calculate the node deltas and weight gradients
        for the connections between network layers.
        Intended for internal use only.

        Parameters
        ----------
        layer_outputs : np.ndarray
            The outputs from the Neural Network layers, as calculated during the feedforward stage.
        target_output : np.ndarray
            The target output (classification values) expected from the network. i.e. the ideal values to be outputted
            by the Output Layer of the Neural Network.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            The calculated node delta values and gradients for each layer in the form (node_deltas, gradients),
           stored under key corresponding to the relevant layer, in the same form as seen in the weights_biases
           attribute.
        """
        node_deltas = {}
        gradients = {}

        # OUTPUT LAYER -> HIDDEN/INPUT
        node_deltas[list(self.weights_biases.keys())[-1]], \
        gradients[list(self.weights_biases.keys())[-1]] = self._backpropagate_output(layer_outputs, target_output)
        # HIDDEN LAYERS -> OTHER
        if len(self.layers) > 2:  # Only need to perform this if there's at least one hidden layer
            # Step backwards through layers, skipping the output layer as we've already done that and the input layer as this has no activation
            for layer_num, layer_output in reversed(list(enumerate(layer_outputs))[1:-1]):
                weight_bias_key = f"{layer_num - 1}_{layer_num}"  # Key for corresponding weights and biases
                next_weight_bias_key = f"{layer_num}_{layer_num + 1}"  # Key for the weights and biases that affect the next layer (in terms of feedforward order)
                layer_inputs = layer_outputs[layer_num - 1]
                node_deltas[weight_bias_key], \
                gradients[weight_bias_key] = self._backpropagate_hidden(
                    self.weights_biases[next_weight_bias_key]["weights"],
                    node_deltas[next_weight_bias_key],
                    layer_inputs,
                    layer_output,
                    self.layers[layer_num]["activation_func"])

        return node_deltas, gradients

    # ref: http://neuralnetworksanddeeplearning.com/chap2.html
    def _backpropagate_output(self, layer_outputs: np.ndarray,
                              target_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate from the Output Layer to a Hidden Layer/Input Layer.
        This function uses matrix operations to improve performance.
        Intended for internal use only.

        Parameters
        ----------
        layer_outputs : np.ndarray
            The outputs from the Neural Network layers, as calculated during the feedforward stage.
        target_output : np.ndarray
            The target output (classification values) expected from the network. i.e. the ideal values to be outputted
            by the Output Layer of the Neural Network.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The calculated node delta values and gradients in the form (node_deltas, gradients).
        """
        dErrorTotal_dLayerOutput = -(target_output - layer_outputs[-1])
        dLayerOutput_dLayerInput = self.layers[-1]["activation_func"](layer_outputs[-1], True)
        delta_output = dLayerOutput_dLayerInput * dErrorTotal_dLayerOutput
        dErrorTotal_dLayerWeight = delta_output @ layer_outputs[-2].T
        return delta_output, dErrorTotal_dLayerWeight

    def _backpropagate_hidden(self, next_weights: np.ndarray, next_node_delta: np.ndarray,
                              layer_input: np.ndarray, layer_output: np.ndarray,
                              layer_activation_func: Callable[[np.ndarray, bool], np.ndarray]
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backpropagate from a Hidden Layer to another layer (i.e. the Input Layer or another Hidden Layer).
        This function uses matrix operations to improve performance.
        Intended for internal use only.

        Parameters
        ----------
        next_weights : np.ndarray
            The connection weights of the current layer and the next layer in the network (in feed forward order).
            i.e If the current layer in the middle layer in a 3 layer network, these weights will be those between
                the current layer (Hidden Layer) and the Output Layer.
        next_node_delta : np.ndarray
            The node deltas of the current layer and the next layer in the network (in feed forward order), calculated
            during backpropagation.
            i.e If the current layer is the middle layer in a 3 layer network, these node deltas will be those between
                the current layer (Hidden Layer) and the Output Layer that were calculated previously during
                backpropagation for the Outer Layer to the Hidden Layer.
        layer_input : np.ndarray
            This input values to the current Neural Network layer.
        layer_output : np.ndarray
            The output (as calculated in the feedforward stage) of the current layer.
        layer_activation_func : Callable[[np.ndarray, bool], np.ndarray]
            The activation function of the current layer.
        Returns
        -------
            Tuple[np.ndarray, np.ndarray]
                The calculated node delta values and gradients in the form (node_deltas, gradients).
        """
        dErrorTotal_dLayerOutput = next_weights.T @ next_node_delta
        dLayerOutput_dLayerInput = layer_activation_func(layer_output, True)
        delta = dLayerOutput_dLayerInput * dErrorTotal_dLayerOutput
        dErrorTotal_dLayerWeight = delta @ layer_input.T
        return delta, dErrorTotal_dLayerWeight

    # TODO - Store the weight and bias update stats
    # TODO - Clean up docs
    # NOTE: The returned bias store isn't currently used for anything.
    def _update_weights_biases(self, learning_rate: float, node_deltas: Dict[str, np.ndarray],
                               gradients: Dict[str, np.ndarray], batch_size: int,
                               l1_lambda: float, return_stats=False
                               ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Update the weights and biases of the Neural Network.
        Intended for internal use only.

        Parameters
        ----------
        learning_rate : float
            The specified learning rate when training the Neural Network.
        node_deltas : Dict[str, np.ndarray]
            The node deltas calculated during backpropagation.
        gradients : Dict[str, np.ndarray]
            The weight gradients calculated during backpropagation.
        batch_size : int
            The size of the batches used during training.
        l1_lambda : float
            The value of the L1 regularisation penalty factor.
        return_stats : bool
            Whether or not to store the weight and bias updates for analysis. If false, return dictionaries
            will be empty. Default is False.
        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            A Tuple of dictionaries containing the calculated weight updates and biases, stored under key corresponding
            to the relevant layer, in the same form as seen in the weights_biases attribute.
            If `return_stats` is False, the returned Tuple will contain empty Dictionaries.
        """
        weight_updates_store = {}
        bias_updates_store = {}
        for key, layer_weights_biases in self.weights_biases.items():
            # Update WEIGHTS
            weight_update_term = (-learning_rate * gradients[key] / batch_size)
            l1_reg_term = self._l1_regularisation(l1_lambda, derivative=True,
                                                  layer_weights=layer_weights_biases["weights"]) if (
                        l1_lambda != 0) else 0
            weight_update = weight_update_term - (learning_rate * l1_reg_term)
            layer_weights_biases["weights"] += weight_update
            # Update BIASES
            bias_update = (-learning_rate * node_deltas[key].sum(axis=1) / batch_size)
            layer_weights_biases["biases"] += bias_update
            # Store update status
            if return_stats:
                weight_updates_store[key] = weight_update
                bias_updates_store[key] = bias_update

        return weight_updates_store, bias_updates_store

    def _l1_regularisation(self, l1_lambda: float, derivative: bool = False,
                           layer_weights: np.ndarray = None) -> Union[float, np.ndarray]:
        """
        Calculate the L1 Regularisation term either for weight updating or error.
        Parameters
        ----------
        l1_lambda : float
        derivative : bool
            Whether or not to calculate using the derivative. This is used should be used for weight updating in
            _update_weights_biases.
            Default is False.
        layer_weights : Optional[np.ndarray]
            The layer weights for the layer to be updated. Only required/ used when `derivative` is set to True.

        Returns
        -------
        Union[float, np.ndarray]
            A float representing the L1 Error if `derivative` is False. An np.ndarray containing the L1 Regularisation
            value for each of the layer weights if `derivative` is set to True.
        """
        if derivative:
            return l1_lambda * np.sign(layer_weights)
        else:
            l1_value = 0
            for key, layer_weights_biases in self.weights_biases.items():
                l1_value += np.sum(np.abs(layer_weights_biases["weights"]))
            return l1_lambda * l1_value

    def test(self, test_data: np.ndarray, test_data_labels: np.ndarray) -> float:
        """
        Test the model on testing data and return the obtained classification accuracy.

        Parameters
        ----------
        test_data : np.ndarray
            The data on with to test the model.
        test_data_labels : np.ndarry
            The corresponding labels for the test data.

        Returns
        -------
        float
            The obtained classification accuracy.
        """
        layer_outputs = self._feedforward(test_data.T)
        classification_accuracy = np.sum(
            np.argmax(layer_outputs[-1], axis=0) == np.argmax(test_data_labels.T, axis=0)) / \
                                  test_data.T.shape[1]
        return classification_accuracy
