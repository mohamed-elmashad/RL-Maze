# agent.py

import numpy as np
import math
import random
import generate_maze as gm
import maze_env as maze
import time
import generator as generate
import rendering as render


class Agent:
    def __init__(self, maze_env, num_episodes=50000, debug_mode=0, logging_enabled=False, max_steps=None, render_mode=0):
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

    def get_explore_rate(self, t):
        return max(self.min_explore_rate, min(0.79, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def state_to_index(self, state):
        # indices = []  # Initialize list to store indices for each dimension
        # for dimension_index in range(len(state)):  # Loop through dimensions of the state
        #     current_state = state[dimension_index]  # Get the current state value for the dimension
        #     lower_bound, upper_bound = self.bounds[dimension_index]  # Get lower and upper bounds for the dimension
            
        #     # Check if current state is below or equal to the lower bound
        #     if current_state <= lower_bound:
        #         index = 0  # Set index to 0 if state is below or equal to lower bound
        #     # Check if current state is above or equal to the upper bound
        #     elif current_state >= upper_bound:
        #         index = self.maze_size[dimension_index] - 1  # Set index to last position if state is above or equal to upper bound
        #     else:
        #         # Calculate index based on linear scaling between bounds and maze size
        #         bound_width = upper_bound - lower_bound
        #         offset = (self.maze_size[dimension_index] - 1) * lower_bound / bound_width
        #         scaling = (self.maze_size[dimension_index] - 1) / bound_width
        #         index = int(round(scaling * current_state - offset))
            
        #     indices.append(index)  # Add calculated index to the list of indices

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
            time.sleep(1.5)
            print("\n")

    def log(self):
        return None

    def simulate(self):
        num_streaks = 0
        learning_rate = self.get_learning_rate(0)
        explore_rate = self.get_explore_rate(0)
        self.env.reset()
        
        for episode in range(self.num_episodes):
            self.env.reset()
            print(f'episode: {episode}')

            pos = self.env.get_state()
            state_0 = self.state_to_index(pos)
            # state = self.env.get_state()
            total_reward = 0

            for t in range(self.max_steps):
                action = self.select_action(state_0, explore_rate)
                observation, reward, done, _ = self.env.step(action)
                pos = self.env.get_state()

                state = self.state_to_index(pos)
                total_reward += reward

                best_q = np.amax(self.q_table[state])
                self.q_table[state_0 + (action,)] += learning_rate * (reward + self.discount_factor * best_q - self.q_table[state_0 + (action,)])

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

            

def main():
    x = 7
    y = 7
    
    maze_size = (x, y)
    start = (0, 0)
    end = (x - 1, y - 1)
    seed = 0
    audio = False
    maze_type = "random"

    generator = generate.MazeGenerator(maze_size, maze_type, seed)
    rendering = render.Rendering(x, y, generator.get_maze(), audio=audio)

    print("MAZE: ", generator.get_maze())
    env = maze.MazeEnv(maze_size, start, end, seed, audio_on=audio, rendering=rendering, generator=generator, mode="human")

    env.render()
    # env.play()
    # env.close()

    solver = Agent(env, num_episodes=50000, debug_mode=1)
    solver.simulate()

if __name__ == "__main__":
    main()