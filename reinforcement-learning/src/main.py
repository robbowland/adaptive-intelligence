from typing import Tuple
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np

from agent.agent import GridWorldAgent
from agent.policy import EpsilonGreedy
from environment.actions import North, East, South, West
from environment.environment import GridWorld
from textwrap import wrap

# Used to calculate the optimal (shortest) path
def manhattan_distance(point_1: Tuple[int, int], point_2: Tuple[int, int]) -> int:
    """
    Calculate the Manhattan Distance between two point.
    This is effectively the distance between two points on a grid
    without diagonal movements.
    This implementation is intended for integer coordinate points.

    Parameters
    ----------
    point_1 : Tuple[int, int]
        The first point.
    point_2 : Tuple[int, int]
        The second point.

    Returns
    -------
    int
        Manhattan Distance between the two points.
    """
    return abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1])


if __name__ == '__main__':
    # CONTROLS
    PRINT_STATS = True
    PRINT_INDIVIDUAL_REPEAT_STATS = False
    PLOT_LEARNING_CURVE = False
    PLOT_DIRECTION_MAP = True
    # = CONSTANTS
    # ~ General
    N_REPEATS = 10
    N_TRIALS = 1000
    N_STEPS = 100
    # ~ Agent
    LEARNING_RATE = 0.3
    DISCOUNT_FACTOR = 0.9
    EPSILON = 0.3
    LAMBDA = 0.15  # Set this to None to use standard SARSA vs SARSA lambda
    # ~ Environment
    ACTIONS = [North, East, South, West]
    ENV_HEIGHT_WIDTH = 10
    BASE_GRID = np.zeros((ENV_HEIGHT_WIDTH, ENV_HEIGHT_WIDTH))
    WALLS = True
    REWARD_LOCATION = (9, 2)
    MOVEMENT_PENALTY = 0.0  # Negative reward applied each action, I didn't end up investigating this, but it is functional
    # Prevent agent from starting on the reward location, or inside an impossible region such as a wall
    SPAWN_LOCATION_BLACKLIST = [
        REWARD_LOCATION,
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (9, 3),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (6, 7),
        (6, 8),
        (6, 9),
    ]

    # = INIT ELEMENTS
    grid = BASE_GRID
    # Set reward location
    grid[REWARD_LOCATION[0], REWARD_LOCATION[1]] = 1.0  # (3,10), (x,y) but uses numpy style indexing
    # Set wall obstacle locations
    if WALLS:
        grid[5:10, 3] = -0.1
        grid[0:4, 5] = -0.1
        grid[6, 7:10] = -0.1

    # ~ Environment
    agent_environment = GridWorld(grid, ACTIONS, out_of_bounds_penalty=-0.1, movement_penalty=MOVEMENT_PENALTY)
    # ~ Agent
    agent_policy = EpsilonGreedy(EPSILON)
    # ~ Stats storage
    repeats_learning_curve = np.zeros((N_REPEATS, N_TRIALS))  # Number of steps before terminal state reached
    repeats_step_excess = np.zeros((N_REPEATS, N_TRIALS))  # Number of steps over the optimum

    # ---- REPEATS
    for repeat_num in range(N_REPEATS):
        print(f"======= Repeat {repeat_num + 1}")
        # Initialise agent with arbitrary start state as this will be changed per trial
        agent = GridWorldAgent(agent_environment, (1, 1), agent_policy, "zeroes", learning_rate=LEARNING_RATE,
                               discount_factor=DISCOUNT_FACTOR)
        # ~ Stats storage
        trial_learning_curve = np.zeros(N_TRIALS)
        trial_step_excess = np.zeros(N_TRIALS)

        # ----- TRIALS
        for trial_num in range(N_TRIALS):
            # ______ Random start for Agent
            agent_start = REWARD_LOCATION  # Give it the reward location to ensure start will be randomised in the next step
            # Ensure the agent does NOT spawn in a blacklisted location, if it does pick a new location
            while agent_start in SPAWN_LOCATION_BLACKLIST:
                agent_start = (np.random.randint(agent_environment.rows), np.random.randint(agent_environment.cols))
            agent.initial_state = agent_start
            agent.reset()

            shortest_path_steps = manhattan_distance(REWARD_LOCATION, agent_start)  # Shortest path from start to reward
            end = False  # Has Agent reached End or exceed N_STEPS?
            step = 0  # Keep track of steps

            # ______ Perform Actions for N_STEPS
            # Let Agent EXPLORE
            while not end and step < N_STEPS:
                step += 1

                if step == N_STEPS:  # Agents has exceeded N_STEPS without finding the end
                    end = agent.act(lambda_=LAMBDA, punish=True)
                else:  # Agent reached the end
                    end = agent.act(lambda_=LAMBDA)

            # Store excess stats
            trial_learning_curve[trial_num] = step
            trial_step_excess[trial_num] = step - shortest_path_steps

        repeats_learning_curve[repeat_num] = trial_learning_curve
        repeats_step_excess[repeat_num] = trial_step_excess

    if PRINT_STATS:
        print(
            f"__________________________\n\
======== SUMMARY =========\n\
__________________________\n\
EXPERIMENT PARAMS \n\
--------------------------\n\
Repeats: {N_REPEATS} \n\
Trials per Repeat: {N_TRIALS}\n\
Max steps per Trial: {N_STEPS} \n\
__________________________\n\
HYPER-PARAMETERS\n\
--------------------------\n\
Learning Rate: {LEARNING_RATE}\n\
Discount Factor: {DISCOUNT_FACTOR}\n\
Epsilon: {EPSILON}\n\
__________________________\n\
STATS\n\
--------------------------"
        )
        # LEARNING CURVE RELATED
        mean_steps = np.mean(repeats_learning_curve, axis=1)
        if PRINT_INDIVIDUAL_REPEAT_STATS:
            print("Average Steps per Repeat:")
            for i in range(N_REPEATS):
                print(f"- Repeat {i + 1}: {mean_steps[i]}")
        # EXCESS STEPS RELATED
        mean_step_excess = np.mean(repeats_step_excess, axis=1)
        mean_steps_excess_all = np.mean(mean_step_excess, axis=0)
        std_step_excess = np.std(repeats_step_excess, axis=1)
        std_step_excess_all = np.std(std_step_excess, axis=0)
        if PRINT_INDIVIDUAL_REPEAT_STATS:
            print("Average Excess Steps per Repeat (with std):")
            for i in range(N_REPEATS):
                print(f"- Repeat{i + 1}: {mean_step_excess[i]} (std: {std_step_excess[i]})")
        print(f"Average Excess Over all Runs: {mean_steps_excess_all}")
        print(f"Average Excess Over all Runs STD: {std_step_excess_all}")
    if PLOT_DIRECTION_MAP:
        # PLOT HEATMAP DIRECTION GRAPH
        agent_q_values = agent.q_table
        max_q_value_grid = np.max(agent_q_values, axis=1).reshape((ENV_HEIGHT_WIDTH, ENV_HEIGHT_WIDTH))
        q_value_directions = np.argmax(agent_q_values, axis=1)
        q_value_directions_grid = q_value_directions.reshape((ENV_HEIGHT_WIDTH, ENV_HEIGHT_WIDTH))
        plt.imshow(max_q_value_grid)
        # plt.imshow(q_value_directions_grid)
        # plt.imshow(agent.environment._grid)
        ARROW_SIZE = 0.5
        # X and Y are flipped on heatmap vs numpy array
        # Heatmap Y is inverted when adding the arrows
        for x in range(0, ENV_HEIGHT_WIDTH):
            for y in range(0, ENV_HEIGHT_WIDTH):
                direction = q_value_directions_grid[y, x]
                # Skip the Terminal states as the run ends there
                if (y, x) in SPAWN_LOCATION_BLACKLIST:
                    continue
                # North
                if direction == 0:
                    dx = 0
                    dy = -ARROW_SIZE
                # East
                if direction == 1:
                    dx = ARROW_SIZE
                    dy = 0
                # South
                if direction == 2:
                    dx = 0
                    dy = ARROW_SIZE
                # West
                if direction == 3:
                    dx = -ARROW_SIZE
                    dy = 0
                plt.arrow(x, y, dx, dy, fc="k", ec="k", head_width=0.1, head_length=0.1)
        plt.colorbar()
        plt.show()

    # Adapted from Lab Code
    if PLOT_LEARNING_CURVE:
        # # Plot some stuff
        means = np.mean(repeats_learning_curve, axis=0)
        errors = np.std(repeats_learning_curve, axis=0) / np.sqrt(
            N_REPEATS)  # errorbars are equal to twice standard error i.e. std/sqrt(samples)

        smooth_means = gaussian_filter1d(means, 2)
        smooth_errors = gaussian_filter1d(errors, 2)

        plt.errorbar(np.arange(N_TRIALS), smooth_means, smooth_errors, 0, elinewidth=0.1, capsize=1, alpha=0.2)
        plt.plot(smooth_means, 'tab:blue')  # Plot the mean on top to standout
        if LAMBDA:
            plt.title("\n".join(wrap(f"Average Steps per Trial using Hyper-Parameters: α = {LEARNING_RATE}, γ = {DISCOUNT_FACTOR}, ε = {EPSILON}, λ = {LAMBDA}", 45)), fontsize=16)
        else:
            plt.title("\n".join(wrap(f"Average Steps per Trial using Hyper-Parameters: α = {LEARNING_RATE}, γ = {DISCOUNT_FACTOR}, ε = {EPSILON}", 45)), fontsize=16)
        plt.xlabel('Trial', fontsize=16)
        plt.ylabel('Average Steps', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig('Sarsa.png', dpi=300)
        plt.show()
