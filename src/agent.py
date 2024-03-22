# agent.py

import numpy as np
import math
import random
import csv
import itertools as it
import os
import time



class Agent:
    def __init__(self, maze_env, num_episodes=50000, debug_mode=0, max_steps=None):
        self.env = maze_env
        self.num_episodes = num_episodes
        self.debug_mode = debug_mode
        self.maze_size = tuple((self.env.observation_space.high + np.ones(self.env.observation_space.shape)).astype(int))
        self.max_steps = max_steps or np.prod(self.env.maze_size) * 100
        self.bounds = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.q_table = np.zeros(self.maze_size + (self.env.action_space.n,), dtype=float)
        self.logging_enabled = False
        
        # constants
        self.discount_factor = 0.99
        self.min_explore_rate = 0.001
        self.min_learning_rate = 0.1
        self.decay_type = "log"
        self.decay_factor = np.prod(self.env.maze_size, dtype=float) / 10.0

        self.original_working_dir = os.getcwd()

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
        # mode 0: no debug
        # mode 1: important debug
        # mode 2: all debug
        # mode 3: csv log (fast edit) TODO remove this mode

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
        elif self.debug_mode == 3:
            # csv_row = f"{episode},{t},{action},{state},{reward},{explore_rate},{learning_rate},{next_state},{np.amax(self.q_table[next_state])}"
            csv_row = f"{episode},{t},{action},{state[0]}, {state[1]},{reward},{explore_rate},{learning_rate},{next_state[0]},{next_state[1]},{np.amax(self.q_table[next_state])}"
            print(csv_row)


    def log(self, episode, total_reward, num_steps, title):
        csv_file = f"{title}.csv"
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['Episode', 'Total Reward', 'Num Steps']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({'Episode': episode, 'Total Reward': total_reward, 'Num Steps': num_steps})
            
    def auto_log(self, num_times=10, extensive=False):
        self.logging_enabled = True
        learning_rates = ["log", "exponential"]
        discount_factors = [0.99, 0.995]

        win_rewards = [10, 100, 1000, 1000000]
        step_rewards = [-0.1, -0.01, -1]
        wall_rewards = [-100, -1000, -10000]

        logs_folder = "data"
        attempt_num = 1
        while os.path.exists(os.path.join(logs_folder, f"{self.maze_size}_attempt_{attempt_num}")):
            attempt_num += 1

        attempt_folder = f"{self.maze_size}_attempt_{attempt_num}"


        for run_id in range(1, num_times + 1):
            os.chdir(self.original_working_dir)  # Change back to the original working directory
            run_folder = os.path.join(logs_folder, attempt_folder, f"run_{run_id}")
            os.makedirs(run_folder, exist_ok=True)
            os.chdir(run_folder)

            if extensive:
                # lr = learning rate, df = discount factor, wr = win reward, sr = step reward, wr = wall reward, dy = decay factor
                for lr, df, wr, sr, wr in it.product(learning_rates, discount_factors, win_rewards, step_rewards, wall_rewards):
                    log_name = f"_lr_{lr}_df_{df}_wr_{wr}_sr_{sr}_wr_{wr}"
                    print(f"\n Running learning rate: {lr}, discount factor: {df}, win reward: {wr}, step reward: {sr}, wall reward: {wr}")
                    self.discount_factor = df
                    self.set_decay_type(lr)
                    self.env.set_reward("widn", wr)
                    self.env.set_reward("step", sr)
                    self.env.set_reward("wall", wr)                
                    
                    self.simulate(log_name)
            else:
                for df, winr, wallr in it.product(discount_factors, win_rewards, wall_rewards):
                    log_name = f"_df_{df}_winr_{winr}_wallr_{wallr}"
                    print(f"\n Running discount factor: {df}, win reward: {winr} wall reward: {wallr}")
                    self.discount_factor = df
                    self.env.set_reward("win", winr)
                    self.env.set_reward("wall", wallr)

                    self.simulate(log_name)
                
    def simulate(self, log_name=None):
        num_streaks = 0
        num_timeouts = 0
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
                    # print(f"Episode {episode} finished after {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    if t < np.prod(self.env.maze_size):
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    num_timeouts = 0
                    break
                elif t >= self.max_steps - 1:
                    # print(f"Episode {episode} timed out at {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    num_timeouts += 1
                    break
            
            if np.std(prev_total_reward) < 0.1 and episode > 30:
                print(f'Coneverged at episode {episode}')
                break
            else:
                prev_total_reward.pop(0)
                prev_total_reward.append(total_reward)

            if self.logging_enabled:
                self.log(episode, total_reward, t, log_name)
            
            if num_timeouts > 20:
                print("Timing out")
                break

            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate( episode)

        