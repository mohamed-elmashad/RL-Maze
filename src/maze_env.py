# maze_env.py

from audio import Audio
from maze_generation import MazeGeneration
from movement import Movement
from gameplay import Gameplay
from visualization import Visualization

class MazeEnv(Env):
	def __init__(self, maze_size=(10, 10), start=(0, 0), end=(9, 9), seed=None, audio_on=False, mode="human"):
		self.seed = seed
		self.maze_size = maze_size
		self.start = start
		self.end = end
		self.mode = mode
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

