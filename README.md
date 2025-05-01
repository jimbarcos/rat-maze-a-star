# A* Pathfinding Rat Maze - Group 5

A Pygame-based visualization where a mouse (rat) navigates through a procedurally generated maze to find cheese using the A* pathfinding algorithm. This educational tool demonstrates how A* pathfinding works by visualizing the search process, calculated paths, and scoring metrics.

## Features

- Procedurally generated maze with guaranteed solution path
- Complete A* pathfinding algorithm visualization
- Animated visualization of both the search process and final path
- Adjustable animation speed with pause/resume capability
- Toggle different visualization layers:
  - Layer 1: The base maze (walls and paths)
  - Layer 2: The graph nodes used by the A* algorithm (open/closed sets)
  - Layer 3: The calculated optimal path
  - Layer 4: All scores (master toggle)
  - Layer 5-7: Individual F/G/H score toggles
- Responsive score display that shows larger values when fewer score types are enabled
- Clean UI with information panels showing:
  - Legend explaining all visualization elements
  - Statistics about the current path and search
  - Controls for all available actions
- Animated mouse character that rotates to face its direction of travel
- Pause/resume functionality to examine the algorithm at any step

## Requirements

- Python 3.6+
- Pygame 2.5.2

## Installation

1. Make sure you have Python installed on your system
2. Install the required package:
```
pip install -r requirements.txt
```

## How to Run

Run the game using:
```
python rat_maze.py
```

## Controls

- `1`: Toggle maze visibility
- `2`: Toggle A* graph nodes visibility
- `3`: Toggle calculated path visibility
- `4`: Toggle all scores (master toggle)
- `5`: Toggle F-scores (total cost)
- `6`: Toggle G-scores (distance from start)
- `7`: Toggle H-scores (heuristic to goal)
- `Space`: Start path following animation
- `A`: Start A* search animation
- `P`: Pause/resume animation
- `+/-`: Adjust search animation speed
- `R`: Generate a new random maze

## A* Algorithm Visualization

This simulation visualizes the famous A* pathfinding algorithm, showing:

### Nodes and Path
- **Light Blue circles**: Open nodes (frontier) being considered
- **Light Pink circles**: Closed nodes already evaluated
- **Orange glowing circle**: Current node being processed
- **Yellow lines**: Calculated optimal path

### A* Scores
- **F-score** (yellow background): Total estimated cost (f = g + h)
- **G-score** (green background): Cost from start to current node
- **H-score** (purple background): Heuristic estimate from current node to goal

### How A* Works
The algorithm maintains two sets:
1. **Open Set**: Nodes discovered but not yet evaluated
2. **Closed Set**: Nodes already evaluated

The algorithm prioritizes nodes with the lowest f-score, exploring the most promising paths first. This makes A* both complete and optimal when using an admissible heuristic.

## Project Structure

- `rat_maze.py`: Main application file containing all game logic and visualization code
- `requirements.txt`: List of required Python packages
- `README.md`: This documentation file
