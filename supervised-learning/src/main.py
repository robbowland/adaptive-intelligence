import math
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import os

import numpy as np
from scipy.io import loadmat
from neuralnet.neuralnetmatrix import NeuralNet
from neuralnet.weights import WeightInit
from neuralnet.activation import ActivationFunc
from multiprocessing import Pool


class Experiment(Enum):
    """
    Experiments corresponding to the different questions in the assignment brief.
    """
    PERCEPTRON_WEIGHT_CONVERGENCE = 0  # Q3 - Perceptron Convergence
    SINGLE_LAYER_TEST_ACCURACY = 1     # Q4 - Perceptron Accuracy on Test Data
    L1_LAMBDA_VALIDATIONS = 2          # Q6 - Effect of L1 Lambda values on Validation Error
    VARYING_HIDDEN_LAYER = 3           # Q7 - Effect of Varying No. of Neurons in Hidden Layer
    VARYING_SECOND_HIDDEN_LAYER = 4    # Q8 - Effect of Varying No. of Neurons in Second Hidden Layer


# Change this to change the experiment that is run
EXPERIMENT = Experiment.PERCEPTRON_WEIGHT_CONVERGENCE
# NOTE: You may need to run some of these in a 'piece-wise' manner and stitch the results together to get the
#       sufficient number of runs and data point values, depending on what you computer can handle.


# =======================================| BEGIN: UTILITY FUNCTIONS |===================================================
# These functions are not well documented as they were written quite late on and solely exist to attempt to speed up
# experimental running by facilitating multiprocessing. I'm not entirely sure if this gave much of a performance
# improvement but it was worth a shot. Each function is used in a different experiment.

# For Assignment Question 6
def test_lambda_value(num_runs, nn_architecture, nn_weight_init, nn_train_data, lambda_value):
    data = np.zeros(num_runs)
    print(f"=== Running Tests for λ_1 = {lambda_value} ===")
    for run in range(0, num_runs):
        print(f"-- Test {run + 1}/{num_runs}")
        nn = NeuralNet(nn_architecture, nn_weight_init)
        validation_errors, _, _, _ = nn.train(*nn_train_data,
                                              lambda_value,
                                              (True, False, False, (False, 0.0), False))
        print(f"Final Epoch Validation Error: {validation_errors[-1]}")
        data[run] += validation_errors[-1]
    return data  # return the validation error of the final epoch


# For Assignment question 7
def test_hidden_layer_sizes(num_runs, nn_activation, nn_weight_init, nn_input_dim, nn_output_dim,
                            nn_train_data, nn_test_data, lambda_value, hidden_layer_num):
    data = np.zeros((num_runs,))
    print(f"=== Running Tests for Hidden Layer Neurons = {hidden_layer_num} ===")
    for run in range(0, num_runs):
        print(f"-- Test {run + 1}/{num_runs}, Hidden Layer Neurons = {hidden_layer_num}")
        #  Network architecture
        nn_architecture = [
            {"dimension": nn_input_dim,
             "activation_func": None},
            {"dimension": hidden_layer_num,
             "activation_func": nn_activation},
            {"dimension": nn_output_dim,
             "activation_func": nn_activation}
        ]
        nn = NeuralNet(nn_architecture, nn_weight_init)
        nn.train(*nn_train_data,
                 lambda_value)
        data[run] += nn.test(*nn_test_data)
        print(f"Test {run + 1}/{num_runs}, Hidden Layer Neurons = {hidden_layer_num}, Classification Accuracy: ", data[run] * 100)
    return data


# For Assignment Question 8
def test_second_hidden_layer_sizes(num_runs, nn_activation, nn_weight_init, nn_input_dim, nn_output_dim,
                                   nn_train_data,  nn_test_data, lambda_value, hidden_layer_num):
    data = np.zeros(num_runs)
    print(f"=== Running Tests for Second Hidden Layer Neurons = {hidden_layer_num} ===")
    for run in range(0, num_runs):
        print(f"-- Test {run + 1}/{num_runs}")
        #  Network architecture
        nn_architecture = [
            {"dimension": nn_input_dim,
             "activation_func": None},
            {"dimension": 100,
             "activation_func": nn_activation},
            {"dimension": hidden_layer_num,
             "activation_func": nn_activation},
            {"dimension": nn_output_dim,
             "activation_func": nn_activation}
        ]
        nn = NeuralNet(nn_architecture, nn_weight_init)
        nn.train(*nn_train_data,
                 lambda_value)
        data[run] += nn.test(*nn_test_data)
        print(f"Test {run + 1}/{num_runs}, Second Hidden Layer Neurons = {hidden_layer_num}, Classification Accuracy: ", data[run] * 100)
    return data
# ===========================================| END: UTILITY FUNCTIONS |=================================================


if __name__ == "__main__":
    # ===========================================| BEGIN: LOAD DATA |===================================================
    EMNIST_DATA_SET = loadmat("./data/emnist-letters-1k.mat")
    NUM_LABELS = 26

    # ____ TRAINING ____________________________________________________________________________________________________
    # Load TRAINING data & labels
    TRAINING_DATA = EMNIST_DATA_SET['train_images']
    # Normalise data to be between 0 and 1
    TRAINING_DATA = (TRAINING_DATA / TRAINING_DATA.max())
    TRAINING_LABELS = EMNIST_DATA_SET['train_labels']
    # Create sparse vectors of training labels where the index corresponding to the correct label is 1 & others are 0
    TRAINING_LABEL_VECTORS = np.zeros((TRAINING_LABELS.shape[0], NUM_LABELS))
    for i in range(0, TRAINING_LABELS.shape[0]):
        TRAINING_LABEL_VECTORS[i, TRAINING_LABELS[i].astype(int)] = 1  # TODO - add reference to the lab notebook
    # __________________________________________________________________________________________________________________

    # ____ TESTING _____________________________________________________________________________________________________
    # Load TESTING data & labels
    TESTING_DATA = EMNIST_DATA_SET['test_images']
    TESTING_LABELS = EMNIST_DATA_SET['test_labels']
    # Create sparse vectors of testing labels where the index corresponding to the correct label is 1 & others are 0
    TESTING_LABEL_VECTORS = np.zeros((TESTING_LABELS.shape[0], NUM_LABELS))
    for i in range(0, TESTING_LABELS.shape[0]):
        TESTING_LABEL_VECTORS[i, TESTING_LABELS[i].astype(int)] = 1
    # __________________________________________________________________________________________________________________

    NUM_TRAINING_SAMPLES, IMG_SIZE = TRAINING_DATA.shape
    # ===========================================| END: LOAD DATA |=====================================================

    # =========================================| BEGIN: EXPERIMENTS |===================================================
    # Below is the code for running each of the required experiments needed to generate results for the report.
    # ____ Q3 __________________________________________________________________________________________________________
    # Description:
    #   Train a single layer network (a perceptron) on the data. When your model has converged the average weight
    #   changes should be close to zero. Plot the sum over all the elements of this average weight update
    #   matrix vs epochs to show that your model has converged.
    if EXPERIMENT is Experiment.PERCEPTRON_WEIGHT_CONVERGENCE:
        # Step 1:
        #   Initialise the Network & Training Parameters
        NUM_VALIDATION_SAMPLES = int(NUM_TRAINING_SAMPLES * 0.2)
        NUM_EPOCHS = 400
        BATCH_SIZE = 50
        ACTIVATION_FUNC = ActivationFunc.relu
        WEIGHT_INIT_METHOD = WeightInit.xavier
        L1_PENALTY_LAMBDA = 0.0  # Disable L1 Regularisation
        # Stat Collection Config
        CALCULATE_VALIDATION_ERROR = False
        CALCULATE_VALIDATION_ACCURACY = False
        CALCULATE_TRAINING_ERRORS = False
        AVERAGE_WEIGHT_UPDATES = (True, 0.01)  # Whether or not to calculate them and the value to use for Tau.
        PRINT_TO_TERMINAL = False
        STATS_CONFIG = (CALCULATE_VALIDATION_ERROR, CALCULATE_VALIDATION_ACCURACY, CALCULATE_TRAINING_ERRORS,
                        AVERAGE_WEIGHT_UPDATES, PRINT_TO_TERMINAL)
        #  Network creation
        nn_architecture = [
            {"dimension": IMG_SIZE,
             "activation_func": None},
            {"dimension": NUM_LABELS,
             "activation_func": ACTIVATION_FUNC}
        ]
        LEARNING_RATES = [0.05]  # Can run multiple, didn't have time to fully explore
        WEIGHT_UPDATES = []

        for learning_rate in LEARNING_RATES:
            print("Running Convergence test for Learning Rate: ", learning_rate)
            nn = NeuralNet(nn_architecture, WEIGHT_INIT_METHOD)
            # Step 2:
            #   Perform Network Training and collect Average Weight Updates.
            _, _, _, average_weight_updates = nn.train(TRAINING_DATA, TRAINING_LABEL_VECTORS, NUM_EPOCHS,
                                                       BATCH_SIZE, NUM_VALIDATION_SAMPLES, learning_rate,
                                                       L1_PENALTY_LAMBDA, STATS_CONFIG)
            # Step 3:
            #   Calculate the sum of the average weights per epoch.
            average_weight_update_per_epoch = [average_weight_matrix["0_1"].sum()
                                               for average_weight_matrix in average_weight_updates]
            WEIGHT_UPDATES.append(average_weight_update_per_epoch)
        WEIGHT_UPDATES_DF = pd.DataFrame()
        for idx, weights in enumerate(WEIGHT_UPDATES):
            WEIGHT_UPDATES_DF[f"{LEARNING_RATES[idx]}"] = WEIGHT_UPDATES[idx]
        csv_name = f"q3_{str(LEARNING_RATES)}.csv"
        save_path = os.path.join(".", "data", "experiments", csv_name)
        WEIGHT_UPDATES_DF.to_csv(save_path)
    # __________________________________________________________________________________________________________________

    # ____ Q4 __________________________________________________________________________________________________________
    # Description:
    #   What classification performance do you achieve on the test data set with your single layer network?
    if EXPERIMENT is Experiment.SINGLE_LAYER_TEST_ACCURACY:
        # Step 1:
        #   Initialise Training & Testing Parameters & Configure Network
        NUM_VALIDATION_SAMPLES = int(NUM_TRAINING_SAMPLES * 0.2)
        NUM_EPOCHS = 400
        BATCH_SIZE = 50
        LEARNING_RATE = 0.05
        ACTIVATION_FUNC = ActivationFunc.relu
        WEIGHT_INIT_METHOD = WeightInit.xavier
        L1_PENALTY_LAMBDA = 0.0 # Disable L1 Regularisation
        # Stat Collection Config
        CALCULATE_VALIDATION_ERROR = False
        CALCULATE_VALIDATION_ACCURACY = False
        CALCULATE_TRAINING_ERRORS = False
        AVERAGE_WEIGHT_UPDATES = (False, 0.00)  # Whether or not to calculate them and the value to use for Tau.
        PRINT_TO_TERMINAL = False
        STATS_CONFIG = (CALCULATE_VALIDATION_ERROR, CALCULATE_VALIDATION_ACCURACY, CALCULATE_TRAINING_ERRORS,
                        AVERAGE_WEIGHT_UPDATES, PRINT_TO_TERMINAL)
        #  Network architecture
        nn_architecture = [
            {"dimension": IMG_SIZE,
             "activation_func": None},
            {"dimension": NUM_LABELS,
             "activation_func": ACTIVATION_FUNC}
        ]
        NUM_TEST_RUNS = 5  # Number of times the network will be created, trained and testing accuracy determined.
        # Step 2:
        #   Initialise store for testing accuracies.
        TEST_ACCURACIES = np.zeros((NUM_TEST_RUNS,))
        # Step 3:
        #   Iteratively create, train and test the network.
        for test in range(0, NUM_TEST_RUNS):
            print(f"Test {test + 1}/{NUM_TEST_RUNS}")
            nn = NeuralNet(nn_architecture, WEIGHT_INIT_METHOD)
            nn.train(TRAINING_DATA, TRAINING_LABEL_VECTORS, NUM_EPOCHS,
                     BATCH_SIZE, NUM_VALIDATION_SAMPLES, LEARNING_RATE,
                     L1_PENALTY_LAMBDA, STATS_CONFIG)
            TEST_ACCURACIES[test] += nn.test(TESTING_DATA, TESTING_LABEL_VECTORS)
            print("Classification Accuracy: ", TEST_ACCURACIES[test] * 100)
        print("Average Accuracy: ", np.mean(TEST_ACCURACIES) * 100)
    # __________________________________________________________________________________________________________________

    # ____ Q6 __________________________________________________________________________________________________________
    # Description:
    #   Train your model starting with a single hidden layer of 50 neurons with the ReLU activation function for a
    #   range of penalty strengths (λ1). Plot the final validation error averaged over 5 different runs against
    #   the values of λ1.
    if EXPERIMENT is Experiment.L1_LAMBDA_VALIDATIONS:
        # Step 1:
        #   Initialise Training Parameters & Configure Network
        NUM_VALIDATION_SAMPLES = int(NUM_TRAINING_SAMPLES * 0.2)
        NUM_EPOCHS = 400
        BATCH_SIZE = 50
        LEARNING_RATE = 0.05
        ACTIVATION_FUNC = ActivationFunc.relu
        WEIGHT_INIT_METHOD = WeightInit.xavier
        #  Network architecture
        nn_architecture = [
            {"dimension": IMG_SIZE,
             "activation_func": None},
            {"dimension": 50,
             "activation_func": ACTIVATION_FUNC},
            {"dimension": NUM_LABELS,
             "activation_func": ACTIVATION_FUNC}
        ]
        NN_TRAIN_DATA = (TRAINING_DATA, TRAINING_LABEL_VECTORS, NUM_EPOCHS, BATCH_SIZE, NUM_VALIDATION_SAMPLES, LEARNING_RATE)
        LAMBDA_VALUES = [0, 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8, 3e-9, 3e-10]  # Lambda values to test
        NUM_RUNS_PER_LAMBDA = 5  # Number of times to train and validate for each lambda value
        # Step 2:
        #   Iteratively create and train the network. Uses multiprocessing to speed things up.
        p = Pool()
        func = partial(test_lambda_value, NUM_RUNS_PER_LAMBDA, nn_architecture, WEIGHT_INIT_METHOD, NN_TRAIN_DATA)
        VALIDATION_ERRORS = p.map(func, LAMBDA_VALUES)
        p.close()
        p.join()
        # Step 3:
        #   Calculate stats and store
        df_columns = [f"Run {i}" for i in range(1, NUM_RUNS_PER_LAMBDA + 1)]
        VALIDATION_ERRORS_DF = pd.DataFrame(VALIDATION_ERRORS,
                                            columns=df_columns)
        VALIDATION_ERRORS_DF["Average Error"] = VALIDATION_ERRORS_DF.mean(axis=1)
        VALIDATION_ERRORS_DF["Standard Deviation"] = VALIDATION_ERRORS_DF[df_columns].std(axis=1)
        VALIDATION_ERRORS_DF["Lambda"] = LAMBDA_VALUES
        # Save data to csv
        csv_name = f"q6_nnh50_e{NUM_EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}_r{NUM_RUNS_PER_LAMBDA}.csv"
        save_path = os.path.join(".", "data", "experiments", csv_name)
        VALIDATION_ERRORS_DF.to_csv(save_path)
    # __________________________________________________________________________________________________________________

    # ____ Q7 __________________________________________________________________________________________________________
    # Description:
    # Vary the number of neurons in the hidden layer and measure the percentage of correct classifications (accuracy) on
    # the test data set. Perform statistics over at least 10 different initial conditions. Plot the average testing
    # accuracy (with error bars) against number of neurons and discuss your results.
    if EXPERIMENT is Experiment.VARYING_HIDDEN_LAYER:
        # Step 1:
        #   Initialise Training Parameters & Configure Network
        NUM_VALIDATION_SAMPLES = int(NUM_TRAINING_SAMPLES * 0.2)
        NUM_EPOCHS = 400
        BATCH_SIZE = 50
        LEARNING_RATE = 0.05
        ACTIVATION_FUNC = ActivationFunc.relu
        WEIGHT_INIT_METHOD = WeightInit.xavier
        L1_LAMBDA = 3e-5
        NN_TRAIN_DATA = (TRAINING_DATA, TRAINING_LABEL_VECTORS, NUM_EPOCHS, BATCH_SIZE,
                         NUM_VALIDATION_SAMPLES, LEARNING_RATE)
        NN_TEST_DATA = (TESTING_DATA, TESTING_LABEL_VECTORS)
        # Neuron numbers that the hidden layer will take.
        HIDDEN_LAYER_NEURONS = [50]
        NUM_RUNS_PER_HIDDEN_LAYER_NUM = 10
        # Step 2:
        #   Iteratively create and train & test the network.
        p = Pool()
        func = partial(test_hidden_layer_sizes, NUM_RUNS_PER_HIDDEN_LAYER_NUM, ACTIVATION_FUNC,
                       WEIGHT_INIT_METHOD, IMG_SIZE, NUM_LABELS, NN_TRAIN_DATA, NN_TEST_DATA, L1_LAMBDA)
        TESTING_ACCURACIES = p.map(func, HIDDEN_LAYER_NEURONS)
        p.close()
        p.join()
        # Step 3:
        #   Calculate stats and store
        df_columns = [f"Run {i}" for i in range(1, NUM_RUNS_PER_HIDDEN_LAYER_NUM + 1)]
        TESTING_ACCURACIES_DF = pd.DataFrame(TESTING_ACCURACIES,
                                             columns=df_columns)
        TESTING_ACCURACIES_DF["Average Accuracy"] = TESTING_ACCURACIES_DF.mean(axis=1)
        TESTING_ACCURACIES_DF["Standard Deviation"] = TESTING_ACCURACIES_DF[df_columns].std(axis=1)
        TESTING_ACCURACIES_DF["Hidden Layer Neurons"] = HIDDEN_LAYER_NEURONS
        # Save data to csv
        csv_name = f"q7_e{NUM_EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}_{str(HIDDEN_LAYER_NEURONS)}.csv"
        save_path = os.path.join(".", "data", "experiments", csv_name)
        TESTING_ACCURACIES_DF.to_csv(save_path)
    # __________________________________________________________________________________________________________________

    # ____ Q8 __________________________________________________________________________________________________________
    # Description:
    # Train your model with a second hidden layer of varying size and your first hidden layer fixed with a size of 100
    # neurons. Again, plot the average testing accuracy (with error bars) against the number of neurons in the
    # 2nd hidden layer.
    if EXPERIMENT is Experiment.VARYING_SECOND_HIDDEN_LAYER:
        # Step 1:
        #   Initialise Training Parameters & Configure Network
        NUM_VALIDATION_SAMPLES = int(NUM_TRAINING_SAMPLES * 0.2)
        NUM_EPOCHS = 400
        BATCH_SIZE = 50
        LEARNING_RATE = 0.05
        ACTIVATION_FUNC = ActivationFunc.relu
        WEIGHT_INIT_METHOD = WeightInit.xavier
        L1_LAMBDA = 3e-5
        NN_TRAIN_DATA = (TRAINING_DATA, TRAINING_LABEL_VECTORS, NUM_EPOCHS, BATCH_SIZE,
                         NUM_VALIDATION_SAMPLES, LEARNING_RATE)
        NN_TEST_DATA = (TESTING_DATA, TESTING_LABEL_VECTORS)
        # Neuron numbers that the hidden layer will take.
        HIDDEN_LAYER_NEURONS = [50, 100, 200, 300, 400]
        NUM_RUNS_PER_HIDDEN_LAYER_NUM = 5
        # Step 2:
        #   Iteratively create and train & test the network.
        p = Pool()
        func = partial(test_second_hidden_layer_sizes, NUM_RUNS_PER_HIDDEN_LAYER_NUM, ACTIVATION_FUNC,
                       WEIGHT_INIT_METHOD, IMG_SIZE, NUM_LABELS, NN_TRAIN_DATA, NN_TEST_DATA, L1_LAMBDA)
        TESTING_ACCURACIES = p.map(func, HIDDEN_LAYER_NEURONS)
        p.close()
        p.join()
        # Step 3:
        #   Calculate stats and store
        df_columns = [f"Run {i}" for i in range(1, NUM_RUNS_PER_HIDDEN_LAYER_NUM + 1)]
        TESTING_ACCURACIES_DF = pd.DataFrame(TESTING_ACCURACIES,
                                             columns=df_columns)
        TESTING_ACCURACIES_DF["Average Accuracy"] = TESTING_ACCURACIES_DF.mean(axis=1)
        TESTING_ACCURACIES_DF["Standard Deviation"] = TESTING_ACCURACIES_DF[df_columns].std(axis=1)
        TESTING_ACCURACIES_DF["Hidden Layer Neurons"] = HIDDEN_LAYER_NEURONS
        # Save data to csv
        csv_name = f"q8_e{NUM_EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}.csv"
        save_path = os.path.join(".", "data", "experiments", csv_name)
        TESTING_ACCURACIES_DF.to_csv(save_path)
    # __________________________________________________________________________________________________________________
    # ==========================================| END: EXPERIMENTS |====================================================
