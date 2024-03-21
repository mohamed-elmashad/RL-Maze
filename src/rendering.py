# rendering.py

import pygame
import sys
import numpy as np
import time

# Initialize Pygame
class Rendering:
    def __init__(self, width, height, maze, audio=False):
        pygame.init()
        self.width = width
        self.height = height
        self.square_size = 30

        self.screen = pygame.display.set_mode((self.width * self.square_size, self.height * self.square_size))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.player_image = pygame.image.load("img/gunrock.png")
        self.player_image = pygame.transform.scale(self.player_image, (self.square_size, self.square_size))
        
        pygame.mixer.init()
        self.maze = maze
        self.maze_size = np.size(maze, 0), np.size(maze, 1)
        print("rendering maze x: ", self.maze_size[0], " y: ", self.maze_size[1])
        self.background_music = pygame.mixer.Sound("music/2-02. Driftveil City.wav")
        self.step_sound = pygame.mixer.Sound("music/smw_coin.wav")
        self.end_sound = pygame.mixer.Sound("music/finish.wav")
        self.init_audio(audio)

    def draw(self, player_pos, mode="human"):
        if mode != "gym":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                    
            # get maze size, it is a 2d numpy array
            self.screen.fill((230, 230, 230))
            for row in range(self.maze_size[0]):
                for col in range(self.maze_size[1]):
                    if self.maze[row][col] == 1: # wall
                        pygame.draw.rect(self.screen, (0, 0, 0), (col * self.square_size, row * self.square_size, self.square_size, self.square_size))
                    elif self.maze[row][col] == 2: # end
                        pygame.draw.rect(self.screen, (0, 255, 0), (col * self.square_size, row * self.square_size, self.square_size, self.square_size))

            self.screen.blit(self.player_image, (player_pos[1] * self.square_size, player_pos[0] * self.square_size))
            pygame.display.flip()
            if mode == "gym":
                self.clock.tick(10000)
        self.clock.tick(10000)

    def get_events(self):
        print("GET EVENTS")
        print(pygame.event.get())
        return pygame.event.get()

    def manual_control(self):
        events = pygame.event.get()
        for event in events:       
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return 0
                elif event.key == pygame.K_DOWN:
                    return 1
                elif event.key == pygame.K_LEFT:
                    return 2
                elif event.key == pygame.K_RIGHT:
                    return 3
                else:
                    continue
        return None
    
    # Close the game
    def close(self):
        pygame.quit()
        sys.exit()
    
    def init_audio(self, audio):
        if audio:
           self.background_music.set_volume(0.35)
           self.step_sound.set_volume(0.65)
           self.end_sound.set_volume(0.99)
        else:
            self.background_music.set_volume(0)
            self.step_sound.set_volume(0)
            self.end_sound.set_volume(0)

    def play_audio(self, sound):
        if sound == "step":
            self.step_sound.play()
        elif sound == "end":
            print("END SOUND")
            self.end_sound.play()
            pygame.time.wait(1000)
        elif sound == "background":
            self.background_music.play()