# main.py

from maze_env import MazeEnv
import pygame
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import heapq


if __name__ == "__main__":
	# end = random(x, y)
	y = 44
	x = 24
	end = (x, y)
	
	x2 = np.random.randint(x - 5, x)
	y2 = np.random.randint(y - 5, y)
	end = (x2, y2)
	env = MazeEnv(maze_size=(x, y), start=(0, 0), end=end, audio_on=True, mode="a_star_search")
	env.play()
	env.close()