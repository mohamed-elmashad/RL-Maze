

from gym import Env
from gym.spaces import Discrete, Box
import rendering
import generator
import utils
import numpy as np

class MazeEnv(Env):
    def __init__(self, maze_size=(10, 10), start=(0, 0), end=(9, 9), seed=None, audio_on=False,
                generator = generator.MazeGenerator, rendering = rendering.Rendering, mode="human"):
        self.maze_size = maze_size
        self.start = start
        self.end = (end[1], end[0])

        self.rewards = { "win": 10, "wall": -100, "step": -0.1 }
        self.seed = seed

        self.audio_on = audio_on
        self.mode = mode
        self.generator = generator
        self.maze = self.generator.get_maze()

        self.player_pos = self.start
        self.rendering = rendering
        self.rendering.init_audio(self.audio_on)
        self.num_steps = 0

        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([self.maze_size[1]-1, self.maze_size[0]-1]), dtype=np.float32)

        self.reset()
        print(self.maze)
    
    def reset(self):
        self.state = self.start
        self.done = False
        self.step_count = 0
        self.reward = 0
        if self.mode != "AI":
            self.render()
        if self.mode != "gym":
            self.rendering.play_audio("background")
        return self.state

    def step(self, action):
        if action == 0:  # up
            new_pos = (self.state[0] - 1, self.state[1])
        elif action == 1:  # down
            new_pos = (self.state[0] + 1, self.state[1])
        elif action == 2:  # left
            new_pos = (self.state[0], self.state[1] - 1)
        elif action == 3:  # right
            new_pos = (self.state[0], self.state[1] + 1)

        if utils.is_valid_move(new_pos, self.maze_size, self.maze):
            self.num_steps += 1
            self.state = new_pos
            if self.mode != "AI":
                self.render()

            if self.state == self.end:
                self.done = True
                self.reward = self.rewards["win"]
            else:
                self.reward = self.rewards["step"]
                # self.rendering.play_audio("step")
        else:
            self.reward = self.rewards["wall"]
            done = False

        return self.maze.copy(), self.reward, self.done, {}

    def close(self):
        self.rendering.close()
    
    def play(self):
        while not self.done:
            if self.mode == "human":
                action = self.rendering.manual_control()
            if action is not None:
                _, _, done, _ = self.step(action)

                if done:
                    print("You've reached the end!")
                    self.rendering.play_audio("end")
                    return
      
    def render(self):
        self.rendering.draw(self.state, self.mode)
    
    def get_state(self):
        return np.array(self.state)

    def set_mode(self, mode):
        self.mode = mode
        self.rendering.set_mode(mode)
    
    def set_reward(self, reward="win", value=10):
        self.rewards[reward] = value


        