# gameplay.py

class Gameplay:
	def play(self):
		if self.mode == "human":
			self.manual_play()
		elif self.mode == "a_star_search":
			print("A* Search")
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


    
	def a_star_play(self):
		path = self.a_star_search()
		actions = self.path_to_actions(path)
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return
			for action in actions:
				_, _, done, _ = self.step(action)
				if done:
					print("You reached the end!")
					pygame.time.wait(2000)
					return

