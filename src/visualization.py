# visualization.py

class Visualization:
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


	def close(self):
		pygame.mixer.quit()
		pygame.quit()


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
			self.render()
			if new_pos == self.end:
				self.end_sound.play()
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
