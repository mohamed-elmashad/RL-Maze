# movement.py

class Movement:
	def is_valid_move(self, pos):
		if pos[0] < 0 or pos[0] >= self.maze_size[0] or pos[1] < 0 or pos[1] >= self.maze_size[1]:
			return False
		if self.maze[pos] == 1:  # Wall
			return False
		return True


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
	

	def get_movement_cost(self, current_node, neighbor):
		# Check if neighbor is a wall node (impassable)
		if self.is_valid_move(neighbor):
			return 99  # Set cost to infinity to avoid moving there

		# Otherwise, check the movement direction
		dx, dy = neighbor[0] - current_node[0], neighbor[1] - current_node[1]
		if abs(dx) + abs(dy) == 2:  # Diagonal movement
			return 1.5 # Example: Diagonal movement is slightly costlier
		else:
			return 1  # Regular movement cost		

	def heuristic(self, a, b):
		dx = abs(a[0] - b[0])
		dy = abs(a[1] - b[1])
		return np.sqrt(2) * min(dx, dy) + abs(dx - dy)

	
	def get_neighbors(self, node):
		neighbors = []
		for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
			neighbor = (node[0] + direction[0], node[1] + direction[1])
			if self.is_valid_move(neighbor):
				neighbors.append(neighbor)
		return neighbors

