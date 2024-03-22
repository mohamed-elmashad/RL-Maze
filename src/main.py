# main.py

import maze_env as maze
import generator as generate
import rendering as render
import agent as agent

def main():
    x = 20
    y = 20
    
    maze_size = (x, y)
    start = (0, 0)
    end = (x - 1, y - 1)
    seed = None
    audio = False
    maze_type = "random"
    generator = generate.MazeGenerator(maze_size, maze_type, seed)
    rendering = render.Rendering(x, y, generator.get_maze(), audio=audio)
    env = maze.MazeEnv(maze_size, start, end, seed, audio_on=audio, rendering=rendering, generator=generator, mode="human")
    solver = agent.Agent(env, num_episodes=50000, debug_mode=0)

    print("MAZE: ", generator.get_maze())

    AI = True
    auto_log = False
    
    if auto_log:
        solver.auto_log()
    elif AI:
        solver.simulate()
    else:
        env.render()
        env.play()
        env.close()


if __name__ == "__main__":
    main()
