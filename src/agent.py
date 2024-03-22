# agent.py

import numpy as np
import math
import random



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
        self.decay_factor = np.prod(self.env.maze_size, dtype=float) / 10.0

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))
        
    def get_explore_rate(self, t, decay_type="log"):
        if decay_type == "log":
            return max(self.min_explore_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))
        elif decay_type == "linear":
            return max(self.min_explore_rate, min(1, 1.0 - (t + 1) / self.decay_factor))
        elif decay_type == "exponential":
            return max(self.min_explore_rate, min(0.8, 0.99 ** (t / self.decay_factor)))
        else:
            raise ValueError("Invalid decay type")


    def get_learning_rate(self, t, decay_type="log"):
        if decay_type == "log":
            return max(self.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))
        elif decay_type == "linear":
            return max(self.min_learning_rate, min(1, 1.0 - (t + 1) / self.decay_factor))
        elif decay_type == "exponential":
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


    def simulate(self):
        num_streaks = 0
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        self.env.reset()
        
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
                    print(f"Episode {episode} finished after {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    if t < np.prod(self.env.maze_size) and total_reward > 2:
                        num_streaks += 1
                    else:
                        num_streaks = 0
                    break
                elif t >= self.max_steps - 1:
                    print(f"Episode {episode} timed out at {t} time steps with total reward = {total_reward} (streak {num_streaks}).")
                    break
            
            if num_streaks > 100:
                print(f"Solved after {episode} episodes.")
                break

            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)

        