# main.py

import maze_env as maze
import generator as generate
import rendering as render

def main():
    x = 13
    y = 15
    
    maze_size = (x, y)
    start = (0, 0)
    end = (x - 1, y - 1)
    seed = 0
    audio = True
    maze_type = "random"
    mode = "human"

    generator = generate.MazeGenerator(maze_size, maze_type, seed)
    rendering = render.Rendering(x, y, generator.get_maze(), audio=audio)

    print("MAZE: ", generator.get_maze())
    env = maze.MazeEnv(maze_size, start, end, seed, audio_on=audio, rendering=rendering, generator=generator, mode=mode)

    env.render()
    env.play()
    env.close()

if __name__ == "__main__":
    main()