# Solving Mazes with Reinforcement Learning in Python

## Introduction

Welcome to our project on solving mazes using reinforcement learning in Python! This project explores the application of Q-Learning, a model-free reinforcement learning algorithm, to teach an agent how to navigate through a maze environment efficiently. Our motivation stemmed from the rising prevalence and interdisciplinary applications of machine learning. By using mazes, which have traditionally been used to study animal behavior, we were able to draw intriguing parallels between machine learning and biological learning.

## Repository Structure

- **main.py**: The main script to run the project
- **maze_env.py**: Defines the maze environment
- **generator.py**: Contains the maze generator logic
- **rendering.py**: Handles the visual rendering of the maze
- **agent.py**: Implements the Q-Learning agent
- **requirements.txt**: Lists the Python dependencies for the project

## Getting Started

### Prerequisites

Ensure you have Python 3.6 or later installed. You can check your Python version using:

```
python --version
```

### Installation

1. Clone the repository:

```
git clone https://github.com/mohamed-elmashad/RL-Maze.git
cd RL-Maze
```

2. Install the required Python packages:

```
pip install -r requirements.txt
```

## Running the Project

To run the project, execute:

```
python main.py
```

## Project Details

### Reinforcement Learning Implementation

Our implementation uses a Q-learning algorithm that teaches an agent to navigate through the maze using rewards and penalties. The agent learns through:

- Four possible moves (up, down, left, right)
- State-action pair evaluation
- Reward optimization
- Dynamic visualization of progress

### Technical Components

The project uses several key technologies:

| Component | Purpose |
|-----------|----------|
| NumPy | Array operations and calculations |
| Pygame | Maze visualization |
| Matplotlib | Performance metrics plotting |
| Gymnasium | Reinforcement learning environment |

## Features

- Random maze generation with guaranteed solutions
- Real-time visualization of learning process
- Customizable maze dimensions
- Adjustable learning parameters
- Performance metrics tracking


## License

[MIT](https://choosealicense.com/licenses/mit/)
```
