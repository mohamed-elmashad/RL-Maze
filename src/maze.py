import sys
import numpy as np
import math
import random
import time

import gym
import gym_maze
import generate_maze as gm


def simulate():
    print("enter simulate")
    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    env.reset()
    # Render tha maze
    print("Initial State:")
    # env.render()
    print("DSADAD")

    for episode in range(NUM_EPISODES):
        print(f'episode: {episode}')

        # Reset the environment
        pos = env.get_position()
        print(f'pos: {pos}')

        # the initial state
        state_0 = state_to_bucket(pos)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)
            # execute the action
            osb, reward, done, _ = env.step(action)
            pos = env.get_position()

            # Observe the result
            state = state_to_bucket(pos)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # if env.is_game_over():
            #     sys.exit()

            if done:
                # env.hard_reset()
                print(q_table)
                env.reset()
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break
                env.reset()

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if best_q > 2:
            break

        # Update parameters
        # explore_rate = get_explore_rate(t/100)
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)
        # learning_rate = 0.1


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.79, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

# def get_explore_rate(t):

def state_to_bucket(state):
    # print("state: ", state)

    #     # Check the type of state and the shape
    # print("Type of state:", type(state))
    # print("Shape of state:", state.shape)


    bucket_indices = []

    for i in range(len(state)):
        # print("Type of STATE_BOUNDS[i][0]:", type(STATE_BOUNDS[i][0]))
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indices.append(bucket_index)

        print("bucket_indices: ", bucket_indices)
    return tuple(bucket_indices)



if __name__ == "__main__":
    # Initialize the "maze" environment
    env = gm.MazeEnv(maze_size=(25, 35), start=(0, 0), end=(24, 24), mode="gym")

    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    # print(print(env.observation_space.low, env.observation_space.high))
    
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.1
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 20.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 2
    RENDER_MAZE = True
    ENABLE_RECORDING = False

    '''
    Creating a Q-Table for each state-action pair
    '''
    print("NUM_BUCKETS: ", NUM_BUCKETS)
    print("NUM_ACTIONS: ", NUM_ACTIONS)
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    # generate random q_table
    # for i in range(q_table.shape[0]):
    #     for j in range(q_table.shape[1]):
    #         for k in range(q_table.shape[2]):
    #             q_table[i][j][k] = random.uniform(0, 1)
            

    '''
    Begin simulation
    '''
    recording_folder = "/tmp/maze_q_learning"

    if ENABLE_RECORDING:
        env.monitor.start(recording_folder, force=True)

    print("\n" + "-"*40)
    simulate()

    # if ENABLE_RECORDING:
    #     env.monitor.close()
