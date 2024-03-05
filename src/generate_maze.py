import pygame
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

class MazeEnv(Env):
	def __init__(self, maze_size=(20, 20), start=(0, 0), end=(19, 19)):
		self.maze_size = maze_size
		self.start = start
		self.end = end
		self.action_space = Discrete(4)  # Up, Down, Left, Right
		self.observation_space = Box(low=0, high=1, shape=(maze_size[0], maze_size[1]), dtype=np.uint8)
		self.reward_range = (-1, 1)
		self.metadata = {}
		self.visited = np.zeros(self.maze_size, dtype=bool)
		self.np_random = np.random.RandomState()

		pygame.init()
		self.screen_size = (maze_size[1] * 30, maze_size[0] * 30)
		self.screen = pygame.display.set_mode(self.screen_size)
		pygame.display.set_caption("Maze Environment")
		self.clock = pygame.time.Clock()

		self.reset()

	def reset(self):
		self.player_pos = self.start
		self.maze = np.zeros(self.maze_size)
		self.maze[self.end] = 2  # End position
		self.generate_maze()
		self.render()
		return self.maze.copy(), {}

	def step(self, action):
		if action == 0:  # Up
			new_pos = (self.player_pos[0] - 1, self.player_pos[1])
		elif action == 1:  # Down
			new_pos = (self.player_pos[0] + 1, self.player_pos[1])
		elif action == 2:  # Left
			new_pos = (self.player_pos[0], self.player_pos[1] - 1)
		elif action == 3:  # Right
			new_pos = (self.player_pos[0], self.player_pos[1] + 1)

		if self.is_valid_move(new_pos):
			self.player_pos = new_pos
			self.render()
			if new_pos == self.end:
				reward = 1
				done = True
			else:
				reward = -0.1
				done = False
		else:
			reward = -0.1
			done = False

		return self.maze.copy(), reward, done, {}

	def render(self, mode='human'):
		self.screen.fill((255, 255, 255))
		for row in range(self.maze_size[0]):
			for col in range(self.maze_size[1]):
				if self.maze[row][col] == 1:  # Wall
					pygame.draw.rect(self.screen, (0, 0, 0), (col * 30, row * 30, 30, 30))
				elif self.maze[row][col] == 2:  # End
					pygame.draw.rect(self.screen, (0, 255, 0), (col * 30, row * 30, 30, 30))
		pygame.draw.rect(self.screen, (255, 0, 0), (self.player_pos[1] * 30, self.player_pos[0] * 30, 30, 30))
		pygame.display.flip()
		self.clock.tick(30)

	def close(self):
		pygame.quit()

	def is_valid_move(self, pos):
		if pos[0] < 0 or pos[0] >= self.maze_size[0] or pos[1] < 0 or pos[1] >= self.maze_size[1]:
			return False
		if self.maze[pos] == 1:  # Wall
			return False
		return True

	def generate_maze(self):
		# set all cells as walls except start and end
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
				next_x, next_y = neighbors[self.np_random.randint(0, len(neighbors))]
				self.maze[(x + next_x) // 2, (y + next_y) // 2] = 0
				stack.append((next_x, next_y))
			else:
				stack.pop()
		
		self.maze[self.end] = 2
		# set all cells next to start as non-walls
		self.maze[self.start[0] + 1, self.start[1]] = 0


		# while stack:
		# 	current_cell = stack[-1]
		# 	visited.add(current_cell)
		# 	neighbors = self.get_unvisited_neighbors(current_cell, visited)

		# 	print(neighbors)
		# 	if neighbors:
		# 		next_cell = neighbors[self.np_random.randint(0, len(neighbors))]
		# 		self.remove_wall(current_cell, next_cell)
		# 		stack.append(next_cell)
		# 	else:
		# 		stack.pop()

		# self.maze.fill(1)

	def get_unvisited_neighbors(self, cell, visited):
		neighbors = []
		directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
		for direction in directions:
			neighbor = (cell[0] + direction[0], cell[1] + direction[1])
			if neighbor not in visited and 0 <= neighbor[0] < self.maze_size[0] and 0 <= neighbor[1] < self.maze_size[1]:
				neighbors.append(neighbor)
		return neighbors

	def remove_wall(self, current_cell, next_cell):
		dx = next_cell[0] - current_cell[0]
		dy = next_cell[1] - current_cell[1]
		wall = ((current_cell[0] + next_cell[0]) // 2, (current_cell[1] + next_cell[1]) // 2)
		self.maze[wall] = 0  # Open the wall

	def seed(self, seed=None):
		self.np_random, seed = np.random.SeedSequence(seed).spawn(1)
		return [seed]

	def manual_play(self):
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_UP:
						action = 0
					elif event.key == pygame.K_DOWN:
						action = 1
					elif event.key == pygame.K_LEFT:
						action = 2
					elif event.key == pygame.K_RIGHT:
						action = 3
					else:
						continue
					_, _, done, _ = self.step(action)
					if done:
						print("You reached the end!")
						return

if __name__ == "__main__":
	env = MazeEnv()
	env.manual_play()
	env.close()