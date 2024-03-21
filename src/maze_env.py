

from gym import Env
from gym.spaces import Discrete, Box
import rendering
import generator
import utils

class MazeEnv(Env):
    def __init__(self, maze_size=(10, 10), start=(0, 0), end=(9, 9), seed=None, audio_on=False,
                generator = generator.MazeGenerator, rendering = rendering.Rendering, mode="human"):
        self.maze_size = maze_size
        self.start = start
        self.end = end
        self.seed = seed

        self.audio_on = audio_on
        self.mode = mode
        self.generator = generator
        self.maze = self.generator.get_maze()

        self.player_pos = self.start
        self.rendering = rendering
        self.num_steps = 0

        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(self.maze_size[0], self.maze_size[1]), dtype=int)

        self.reset()
        print(self.maze)
    
    def reset(self):
        self.state = self.start
        self.done = False
        self.step_count = 0
        self.reward = 0
        self.render()
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
            print("player_pos:", self.state)
            self.render()
            if self.state == self.end:
                self.done = True
                self.reward = 10
            else:
                self.reward = -0.1
        else:
            self.reward = -100

        # self.render()
        return self.maze.copy(), self.reward, self.done, {}

    def close(self):
        self.rendering.close()
    
    def play(self):
        while not self.done:
        # while True:
            action = self.rendering.manual_control()
            if action is not None:
                print("ACTION: ", action)

            # self.render()
                _, _, done, _ = self.step(action)

                if done:
                    print("You've reached the end!")

      
    def render(self):
        self.rendering.draw(self.state, self.mode)
        # return self.rendering.get_events()
    


        