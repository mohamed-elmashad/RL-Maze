# pathfinding.py

import heapq
import utils
import numpy as np

def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    np.sqrt(2) * min(dx, dy) + abs(dx - dy)

def get_neighbors(node):
    neighbors = []
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_node = (node[0] + direction[0], node[1] + direction[1])
        if new_node[0] >= 0 and new_node[0] < 10 and new_node[1] >= 0 and new_node[1] < 10:
            if not utils.is_wall(new_node):
                neighbors.append(new_node)
    return neighbors

def movement_cost(current_node, neighbor):
    if utils.is_valid_move(neighbor):
        return 99
    
    dx, dy = neighbor[0] - current_node[0], neighbor[1] - current_node[1]
    if abs(dx) + abs(dy) == 2:
        return 100 # diagonal movement not allowed
    else:
        return 1
    

def a_star_path(start, end):
    open_list = []  # Use a list to maintain the order of elements
    open_dict = {}  # Use a dictionary for faster lookups and updates
    closed_list = {}  # Initialize closed list
    g_costs = {start: 0}  # Initialize g_costs dictionary

    # Initialize start node
    heapq.heappush(open_list, (0, start, None))
    open_dict[start] = 0  # Priority of start node is 0

    while open_list:
        current_node = heapq.heappop(open_list)[1]
        parent = heapq.heappop(open_list)[2]

        # Check if the current node is the goal
        if current_node == end:
            # Reconstruct path and return it
            path = [current_node]
            while parent:
                path.append(parent)
                parent = closed_list[parent]
            return path[::-1]
        
        # Add the current node to the closed list
        closed_list[current_node] = parent

        # Get neighbors of the current node
        unfiltered_neighbors = get_neighbors(current_node)
        neighbors = [neighbor for neighbor in unfiltered_neighbors if utils.is_valid_move(neighbor)]

        for neighbor in neighbors:
            # Calculate the tentative g score for the neighbor
            neighbor_g = g_costs[current_node] + movement_cost(current_node, neighbor)

            # Check if the neighbor is already in the closed list
            if neighbor in closed_list:
                continue

            # Calculate the f score for the neighbor
            neighbor_f = neighbor_g + heuristic(neighbor, end)

            # Check if the neighbor is already in the open list
            if neighbor in open_dict:
                # Update the priority of the neighbor in open_list and open_dict
                if neighbor_g < g_costs[neighbor]:
                    open_list.remove((open_dict[neighbor], neighbor, parent))
                    heapq.heapify(open_list)
                    open_dict[neighbor] = neighbor_g
                    heapq.heappush(open_list, (neighbor_f, neighbor, current_node))
            else:
                # Add the neighbor to open_list and open_dict
                open_dict[neighbor] = neighbor_g
                g_costs[neighbor] = neighbor_g
                heapq.heappush(open_list, (neighbor_f, neighbor, current_node))
    return None # No path found


