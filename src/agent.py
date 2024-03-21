# agent.py

import numpy as np
import math
import random
import generate_maze as gm
import maze_env as maze
import rendering as render
import generator as generate

class MazeSolver:
    def __init__(self, maze_env, num_episodes=50000, max_steps=None):
        self.env = maze_env
        self.num_episodes = num_episodes
        self.max_steps = max_steps or np.prod(self.env.maze_size) * 100
        self.q_table = np.zeros(self.env.num_buckets + (self.env.num_actions,), dtype=float)
        self.min_explore_rate = 0.001
        self.min_learning_rate = 0.1
        self.decay_factor = np.prod(self.env.maze_size, dtype=float) / 20.0

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[state]))

    def get_explore_rate(self, t):
        return max(self.min_explore_rate, min(0.79, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, min(0.8, 1.0 - math.log10((t + 1) / self.decay_factor)))

    def state_to_bucket(self, state):
        bucket_indices = []
        for i in range(len(state)):
            if state[i] <= self.env.state_bounds[i][0]:
                bucket_index = 0
            elif state[i] >= self.env.state_bounds[i][1]:
                bucket_index = self.env.num_buckets[i] - 1
            else:
                bound_width = self.env.state_bounds[i][1] - self.env.state_bounds[i][0]
                offset = (self.env.num_buckets[i] - 1) * self.env.state_bounds[i][0] / bound_width
                scaling = (self.env.num_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state[i] - offset))
            bucket_indices.append(bucket_index)
        return tuple(bucket_indices)

    def simulate(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            explore_rate = self.get_explore_rate(episode)
            learning_rate = self.get_learning_rate(episode)
            total_reward = 0

            for t in range(self.max_steps):
                action = self.select_action(state, explore_rate)
                next_state, reward, done, _ = self.env.step(action)

                best_q = np.amax(self.q_table[next_state])
                self.q_table[state + (action,)] += learning_rate * (reward + 0.99 * best_q - self.q_table[state + (action,)])

                state = next_state
                total_reward += reward

                if done:
                    break

            

def main():
    x = 8
    y = 5
    
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
    env.play()
    env.close()

    solver = MazeSolver(env)
    solver.simulate()

if __name__ == "__main__":
    main()