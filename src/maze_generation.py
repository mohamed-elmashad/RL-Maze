# maze_generation.py

class MazeGeneration:
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
