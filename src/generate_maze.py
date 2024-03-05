import pygame
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import heapq

class MazeEnv(Env):
	def __init__(self, maze_size=(10, 10), start=(0, 0), end=(9, 9), seed=None, audio_on=False, mode="human"):
		self.seed = seed
		self.maze_size = maze_size
		self.start = start
		self.end = end
		self.mode = mode
		self.action_space = Discrete(4)  # Up, Down, Left, Right
		self.observation_space = Box(low=0, high=1, shape=(maze_size[0], maze_size[1]), dtype=np.uint8)
		self.reward_range = (-1, 1)
		self.metadata = {}
		self.visited = np.zeros(self.maze_size, dtype=bool)
		# self.np_random = np.random.RandomState()
		self.np_random = np.random.RandomState(seed=self.seed) 

		# initialize pygame mixer
		self.audio_on = audio_on
		pygame.mixer.init()
		self.background_music = pygame.mixer.Sound("music/2-02. Driftveil City.wav")
		self.step_sound = pygame.mixer.Sound("music/smw_coin.wav")
		self.end_sound = pygame.mixer.Sound("music/finish.wav")

		self.player_image = pygame.image.load("img/giang-2.png")
		self.player_image = pygame.transform.scale(self.player_image, (30, 30))  # Resize the image to fit the cell size

		pygame.init()
		self.screen_size = (maze_size[1] * 30, maze_size[0] * 30)
		self.screen = pygame.display.set_mode(self.screen_size)
		pygame.display.set_caption("Maze Environment")
		self.clock = pygame.time.Clock()

		self.init_audio()

		self.reset()

	def init_audio(self):
		if self.audio_on:
			self.background_music.set_volume(0.35)
			self.step_sound.set_volume(0.6)
			self.end_sound.set_volume(0.99)
		else: # Turn off audio
			self.background_music.set_volume(0)
			self.step_sound.set_volume(0)
			self.end_sound.set_volume(0)
		

	# def reset(self):
		# self.player_pos = self.start
		# self.maze = np.zeros(self.maze_size)
		# self.maze[self.end] = 2  # End position
		# self.generate_maze()
		# self.render()
		# self.background_music.play()
		# path = self.a_star_search()

		# return self.maze.copy(), {"path": path}

	def reset(self):
		self.player_pos = self.start
		self.maze = np.zeros(self.maze_size)
		self.maze[self.end] = 2  # End position
		self.generate_maze()
		self.background_music.play()
		self.render()  # Pass the path coordinates to render
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
			print("TEST")
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
		self.step_sound.play()

		return self.maze.copy(), reward, done, {}

	def render(self):
		self.screen.fill((230, 230, 230))
		for row in range(self.maze_size[0]):
			for col in range(self.maze_size[1]):
				if self.maze[row][col] == 1:  # Wall
					pygame.draw.rect(self.screen, (0, 0, 0), (col * 30, row * 30, 30, 30))
				elif self.maze[row][col] == 2:  # End
					pygame.draw.rect(self.screen, (0, 255, 0), (col * 30, row * 30, 30, 30))
		# pygame.draw.rect(self.screen, (255, 0, 0), (self.player_pos[1] * 30, self.player_pos[0] * 30, 30, 30))
		self.screen.blit(self.player_image, (self.player_pos[1] * 30, self.player_pos[0] * 30))
		pygame.display.flip()
		self.clock.tick(30)

	#  def render(self, explored=None, path=None, mode='human'):
    #     if mode == 'human':
    #         # Existing human rendering code
    #     elif mode == 'a_star':
    #         self.render_a_star(explored, path)  # Call a new function for A* visualization
    #     else:
    #         raise ValueError(f"Unknown rendering mode: {mode}")

	# def render(self, explored=None, path=None, mode="human"):
	# 	self.screen.fill((230, 230, 230))  # Fill background

	# 	print("Mode: ", mode)
	# 	# Draw maze elements
	# 	for row in range(self.maze_size[0]):
	# 		for col in range(self.maze_size[1]):
	# 			if self.maze[row][col] == 1:  # Wall
	# 				pygame.draw.rect(self.screen, (0, 0, 0), (col * 30, row * 30, 30, 30))
	# 			elif self.maze[row][col] == 2:  # End
	# 				pygame.draw.rect(self.screen, (0, 255, 0), (col * 30, row * 30, 30, 30))

	# 	# Draw player
	# 	self.screen.blit(self.player_image, (self.player_pos[1] * 30, self.player_pos[0] * 30))

	# 	print("Mode: ", mode)
	# 	if mode == "a_star" and explored is not None and path is not None:
	# 		for pos in explored:
	# 			pygame.draw.rect(self.screen, (128, 128, 128), (pos[1] * 30, pos[0] * 30, 30, 30))  # Explored nodes
	# 		for pos in path:
	# 			pygame.draw.rect(self.screen, (255, 165, 0), (pos[1] * 30, pos[0] * 30, 30, 30))  # Path nodes

	# 	# Update display and clock
	# 	pygame.display.flip()
	# 	self.clock.tick(30)

	def close(self):
		pygame.mixer.quit()
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
	
	def play(self):
		if self.mode == "human":
			self.manual_play()
		elif self.mode == "a_star_search":
			self.a_star_play()
		else:
			raise ValueError(f"Unknown mode: {self.mode}")

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
						# wait 10 seconds then return
						pygame.time.wait(2000)
						return
		
		# def a_star_play(self):
		# 	path = self.a_star_search()
		# 	actions = self.path_to_actions()
		# 	for action in actions:
		# 		self.player_pos = pos
		# 		self.render()
		# 		pygame.time.wait(100)
		
		# def path_to_actions(self, path):
		# 	actions = []
		# 	# Convert path to actions

		# def a_star_search(self):
		# 	# A* search algorithm that returns the path from start to end without hitting any walls

	# def a_star_search(self):
	# 	start = self.start
	# 	end = self.end

	# 	open_list = []  # Use a list to maintain the order of elements
	# 	open_dict = {}  # Use a dictionary for faster lookups and updates

	# 	# Initialize start node
	# 	heapq.heappush(open_list, (0, start, None))
	# 	open_dict[start] = 0  # Priority of start node is 0

	# 	# Inside the loop where you update the priority of a node:
	# 	if neighbor in open_dict:
	# 		# Update the priority of the neighbor in open_list and open_dict
	# 		open_list.remove((open_dict[neighbor], neighbor, parent))
	# 		open_dict[neighbor] = new_priority
	# 		heapq.heappush(open_list, (new_priority, neighbor, parent))
	# 	else:
	# 		# Add the neighbor to open_list and open_dict
	# 		open_dict[neighbor] = new_priority
	# 		heapq.heappush(open_list, (new_priority, neighbor, parent))
	# 		closed_list = set()  # Use a set for efficient membership checks

	# 	# Initialize start node
	# 	heapq.heappush(open_list, (0, start, None))

	# 	g_costs = {start: 0}
	# 	f_costs = {start: self.heuristic(start, end)}
		

	# 	while open_list:
	# 		current_f, current_node, parent = heapq.heappop(open_list)
	# 		print(current_f, current_node, parent)
	# 		# Check if the current node is the goal
	# 		if current_node == end:
	# 			# Reconstruct path and return it
	# 			path = [current_node]
	# 			while parent:
	# 				path.append(parent)
	# 				parent = closed_list[parent]
	# 			return path[::-1]

	# 		# Add the current node to the closed list
	# 		closed_list.add(current_node)

	# 		# Get neighbors of the current node
	# 		unfiltered_neighbors = self.get_neighbors(current_node)
	# 		neighbors = [neighbor for neighbor in unfiltered_neighbors if self.is_valid_move(neighbor)]

	# 		for neighbor in neighbors:
	# 			# Calculate the tentative g score for the neighbor
	# 			neighbor_g = g_costs[current_node] + self.get_movement_cost(current_node, neighbor)
	# 			print(neighbor_g)

	# 			# Check if the neighbor is already in the open list
	# 			if neighbor in open_list:
	# 				# Calculate the f score for the neighbor using the existing g score
	# 				neighbor_f = neighbor_g + f_costs[neighbor]
	# 			else:
	# 				# Initialize the g and f scores for the neighbor
	# 				g_costs[neighbor] = neighbor_g
	# 				f_costs[neighbor] = neighbor_g + self.heuristic(neighbor, end)
	# 				heapq.heappush(open_list, (f_costs[neighbor], neighbor, current_node))

	# 			# Update the g and f scores if the new g score is lower than the current g score
	# 			if neighbor_g < g_costs[neighbor]:
	# 				g_costs[neighbor] = neighbor_g
	# 				f_costs[neighbor] = neighbor_g + self.heuristic(neighbor, end)
	# 				heapq.heappush(open_list, (f_costs[neighbor], neighbor, current_node))

	# 	return []  # No path found	

	def a_star_search(self):
		start = self.start
		end = self.end

		open_list = []  # Use a list to maintain the order of elements
		open_dict = {}  # Use a dictionary for faster lookups and updates
		closed_list = {}  # Initialize closed list
		g_costs = {start: 0}  # Initialize g_costs dictionary


		# Initialize start node
		heapq.heappush(open_list, (0, start, None))
		open_dict[start] = 0  # Priority of start node is 0

		while open_list:
			current_f, current_node, parent = heapq.heappop(open_list)

			# Check if the current node is the goal
			if current_node == end:
				# Reconstruct path and return it
				path = [current_node]
				while parent:
					path.append(parent)
					parent = closed_list[parent]
				return path[::-1]

			# Add the current node to the closed list
			closed_list[current_node] = parent

			# Get neighbors of the current node
			unfiltered_neighbors = self.get_neighbors(current_node)
			neighbors = [neighbor for neighbor in unfiltered_neighbors if self.is_valid_move(neighbor)]

			for neighbor in neighbors:
				# Calculate the tentative g score for the neighbor
				neighbor_g = g_costs[current_node] + self.get_movement_cost(current_node, neighbor)

				# Check if the neighbor is already in the closed list
				if neighbor in closed_list:
					continue

				# Calculate the f score for the neighbor
				neighbor_f = neighbor_g + self.heuristic(neighbor, end)

				# Check if the neighbor is already in the open list
				if neighbor in open_dict:
					# Update the priority of the neighbor in open_list and open_dict
					if neighbor_g < g_costs[neighbor]:
						open_list.remove((open_dict[neighbor], neighbor, parent))
						heapq.heapify(open_list)
						open_dict[neighbor] = neighbor_g
						heapq.heappush(open_list, (neighbor_f, neighbor, current_node))
				else:
					# Add the neighbor to open_list and open_dict
					open_dict[neighbor] = neighbor_g
					g_costs[neighbor] = neighbor_g  # Update g_costs for the neighbor
					heapq.heappush(open_list, (neighbor_f, neighbor, current_node))

		return []  # No path found

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

	def path_to_actions(self, path):
		actions = []
		for i in range(len(path) - 1):
			current_pos = path[i]
			next_pos = path[i + 1]
			dx = next_pos[0] - current_pos[0]
			dy = next_pos[1] - current_pos[1]

			if dx == 1:
				actions.append(1)  # Down
			elif dx == -1:
				actions.append(0)  # Up
			elif dy == 1:
				actions.append(3)  # Right
			elif dy == -1:
				actions.append(2)  # Left

		return actions

	def a_star_play(self):
		path = self.a_star_search()
		print(path)
		actions = self.path_to_actions(path)
		for action in actions:
			print(action)
			_, _, done, _ = self.step(action)
			if done:
				print("You reached the end!")
				pygame.time.wait(2000)
				return





if __name__ == "__main__":
	# end = random(x, y)
	y = 12
	x = 12
	end = (x, y)
	
	x2 = np.random.randint(x - 5, x)
	y2 = np.random.randint(y - 5, y)
	end = (x2, y2)
	env = MazeEnv(maze_size=(x, y), start=(0, 0), end=end, audio_on=False, mode="a_star_search")
	env.play()
	env.close()