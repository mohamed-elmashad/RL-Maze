# agent.py

import numpy as np
import math
import random
import csv
import itertools as it
import os
import time



class Agent:
    def __init__(self, maze_env, num_episodes=50000, debug_mode=0, logging_enabled=False, max_steps=None):
        self.env = maze_env
        self.num_episodes = num_episodes
        self.debug_mode = debug_mode
        self.logging_enabled = logging_enabled
        self.maze_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))
        self.max_steps = max_steps or np.prod(self.env.maze_size) * 100
        self.bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.q_table = np.zeros(self.maze_size + (self.env.action_space.n,), dtype=float)
        
        # constants
        self.discount_factor = 0.99
        self.min_explore_rate = 0.001
        self.min_learning_rate = 0.1
        self.decay_type = "log"
        self.decay_factor = np.prod(self.env.maze_size, dtype=float) / 10.0

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
        
    def set_decay_type(self, decay_type):
        self.decay_type = decay_type
        
    def get_explore_rate(self, t):
        if self.decay_type == "log":
            return max(self.min_explore_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))
        elif self.decay_type == "linear":
            return max(self.min_explore_rate, min(1, 1.0 - (t + 1) / self.decay_factor))
        elif self.decay_type == "exponential":
            return max(self.min_explore_rate, min(0.8, 0.99 ** (t / self.decay_factor)))
        else:
            raise ValueError("Invalid decay type")


    def get_learning_rate(self, t):
        if self.decay_type == "log":
            return max(self.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))
        elif self.decay_type == "linear":
            return max(self.min_learning_rate, min(1, 1.0 - (t + 1) / self.decay_factor))
        elif self.decay_type == "exponential":
            return max(self.min_learning_rate, min(0.8, 0.99 ** (t / self.decay_factor)))
        else:
            raise ValueError("Invalid decay type")

    def state_to_index(self, state):
        indices = []
        for i in range(len(state)):
            if state[i] <= self.bounds[i][0]:
                index = 0
            elif state[i] >= self.bounds[i][1]:
                index = self.maze_size[i] - 1
            else:
                bound_width = self.bounds[i][1] - self.bounds[i][0]
                offset = (self.maze_size[i] - 1) * self.bounds[i][0] / bound_width
                scaling = (self.maze_size[i] - 1) / bound_width
                index = int(round(scaling * state[i] - offset))
            indices.append(index)
        
        return tuple(indices)  # Return tuple of indices for each dimension

    def debug(self, episode, t, explore_rate, learning_rate, state, action, reward, next_state):
        if self.debug_mode == 1:
            print("\nEpisode: ", episode)
            print("Timestep: ", t)
            print("Action: ", action)
            print("State: ", state)
            print("Reward: ", reward)
            print("Explore rate: ", explore_rate)
            print("Learning rate: ", learning_rate)
            print("Next state: ", next_state)
            print("Best Q: ", np.amax(self.q_table[next_state]))
            print("\n")

           
        elif self.debug_mode == 2:
            print("\nEpisode: ", episode)
            print("Timestep: ", t)
            print("Action: ", action)
            print("State: ", state)
            print("Reward: ", reward)
            print("Explore rate: ", explore_rate)
            print("Learning rate: ", learning_rate)
            print("Next state: ", next_state)
            print("Best Q: ", np.amax(self.q_table[next_state]))
            print("Q-table: ", self.q_table)
            print("\n")


    def log(self, episode, total_reward, num_steps, title):
        csv_file = title + '_log.csv'
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['Episode', 'Total Reward', 'Num Steps']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({'Episode': episode, 'Total Reward': total_reward, 'Num Steps': num_steps})

    def auto_log(self, num_times=10):
        learning_rates = ["log", "linear", "exponential"]
        discount_factors = [0.99, 0.995, 0.95, 0.9]

        win_rewards = [1, 10, 100, 1000]
        step_rewards = [-0.1, -0.01, -0.001, -0.0001]
        wall_rewards = [-100, -1000, -10000]

        decay_factors = [10, 100, 1000, 10000]


        logs_folder = "data"
        for run_id in range(num_times):
            run_folder = os.path.join(logs_folder, f"run_{run_id}")
            os.makedirs(run_folder, exist_ok=True)
            os.chdir(run_folder)
            time.sleep(60)
        
            # lr = learning rate, df = discount factor, wr = win reward, sr = step reward, wr = wall reward, dy = decay factor
            for lr, df, wr, sr, wr, dy in it.product(learning_rates, discount_factors, win_rewards, step_rewards, wall_rewards, decay_factors):
                log_name = f"maze_{self.maze_size}_lr_{lr}_df_{df}_wr_{wr}_sr_{sr}_wr_{wr}_dy_{dy}"
                self.discount_factor = df
                self.set_decay_type(lr)
                self.env.set_reward("win", wr)
                self.env.set_reward("step", sr)
                self.env.set_reward("wall", wr)                
                
                self.decay_factor = dy
                self.simulate()
                self.log(log_name)

    def simulate(self):
        num_streaks = 0
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        self.env.reset()

        prev_total_reward = [0] * 30

        for episode in range(self.num_episodes):
            self.env.reset()

            pos = self.env.get_state()
            state_0 = self.state_to_index(pos)
            total_reward = 0

            for t in range(self.max_steps):
                action = self.select_action(state_0, explore_rate)
                observation, reward, done, _ = self.env.step(action)
                pos = self.env.get_state()

                state = self.state_to_index(pos)
                total_reward += reward

                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[state_0 + (action,)])

                self.debug(episode, t, explore_rate, learning_rate, state, action, reward, state_0)
                state_0 = state

                if done:
                    if self.logging_enabled:
                        self.log(episode, total_reward, t, "q_learning")
                    print(f"Episode {episode} finished after {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    if t < np.prod(self.env.maze_size):
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break
                elif t >= self.max_steps - 1:
                    print(f"Episode {episode} timed out at {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    break
            
            if abs(total_reward - np.mean(prev_total_reward)) < 0.1:
                print(f'Coneverged at episode {episode}')
                break
            else:
                prev_total_reward.pop(0)
                prev_total_reward.append(total_reward)

            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)

        