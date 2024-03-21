# generator.py

import random
import numpy as np

class MazeGenerator:
    def __init__(self, maze_size, maze_type="random", seed=0):
        self.maze_size = maze_size
        self.seed = seed
        self.seedRandom = np.random.RandomState(seed)
        self.maze = np.zeros(self.maze_size)
        self.start = (0, 0)
        self.end = (self.maze_size[0] - 1, self.maze_size[1] - 1)
        self.maze_type = maze_type
        self.generate_maze()

    def reset(self):
        self.maze = np.zeros(self.maze_size)
        self.start = (0, 0)
        self.end = (self.maze_size[0] - 1, self.maze_size[1] - 1)
        self.generate_maze()
        return self.maze, self.start, self.end

    def generate_maze(self):
        if self.maze_type == "random":
            self.random_maze()
        elif self.maze_type == "empty":
            self.empty_maze()
        else:
            raise ValueError("Invalid maze type")
        
    def random_maze(self):
        self.maze.fill(1)
        self.maze[self.start] = 0
        stack = [self.end]

        while stack:
            x, y = stack[-1]
            self.maze[x, y] = 0
            neighbors = []
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + 2 * direction[0], y + 2 * direction[1]
                if 0 <= nx < self.maze_size[0] and 0 <= ny < self.maze_size[1] and self.maze[nx, ny] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                if self.seed != None:
                    next_x, next_y = neighbors[self.seedRandom.randint(0, len(neighbors))]
                else:
                    next_x, next_y = neighbors[random.randint(0, len(neighbors) - 1)]
                self.maze[(x + next_x) // 2, (y + next_y) // 2] = 0
                stack.append((next_x, next_y))
            else:
                stack.pop()

        self.maze[self.end] = 2
        self.maze[self.start[0] + 1, self.start[1]] = 0
        # transpose maze (TODO figure our why the maze is transposed in the first place)
        self.maze = np.transpose(self.maze)

    def empty_maze(self):
        self.maze.fill(0)
        self.maze[self.start] = 0
        self.maze[self.end] = 2
        self.maze[self.start[0] + 1, self.start[1]] = 0

    def get_maze(self):
        return self.maze




