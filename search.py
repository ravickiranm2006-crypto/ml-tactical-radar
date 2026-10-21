"""
Classical Search Algorithms for Stealth Navigation:
Space-Time Breadth-First Search (BFS) and Space-Time A* Search.

Academic Justification:
In environments with dynamic deterministic obstacles (moving guards), planning on a 
static 2D snapshot is flawed and collision-prone. Space-Time search represents states as
(row, col, timestep). At each step, transitioning from (r, c, t) to (r', c', t+1) must not
enter the detection zone of guards at time t+1. This is the standard, academically 
rigorous formulation for dynamic obstacle pathfinding (Silver 2005, Space-Time A*).
"""

import time
import heapq
from collections import deque

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def space_time_bfs(env, initial_phase=0):
    """
    Breadth-First Search on the Space-Time graph (r, c, t).
    Guarantees the shortest path in timesteps to reach target without detection.
    
    Returns:
        dict containing:
            'path': list of (r, c) positions from start to target
            'actions': list of actions taken
            'nodes_expanded': number of expanded search states
            'time_taken': computation time in seconds
            'success': True if target reached, False otherwise
    """
    start_time = time.perf_counter()
    start_pos = env.start
    target_pos = env.target
    max_steps = env.max_steps

    # Queue stores: (current_pos, t, path_positions, actions)
    start_state = (start_pos, 0)
    queue = deque([(start_pos, 0, [start_pos], [])])
    visited = {start_state}
    nodes_expanded = 0

    while queue:
        curr_pos, t, path, actions = queue.popleft()
        nodes_expanded += 1

        if curr_pos == target_pos:
            return {
                'path': path,
                'actions': actions,
                'nodes_expanded': nodes_expanded,
                'time_taken': time.perf_counter() - start_time,
                'success': True,
                'cost': len(actions)
            }

        if t >= max_steps:
            continue

        # Try all 5 actions: Up, Down, Left, Right, Stay
        for action_idx, (dr, dc) in env.ACTIONS.items():
            next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            next_t = t + 1

            # Check boundary and static obstacles
            if not env.is_valid_cell(next_pos):
                continue

            # Check dynamic detection zone at t+1
            if env.is_detected(next_pos, t=next_t + initial_phase):
                continue

            next_state = (next_pos, next_t)
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_pos, next_t, path + [next_pos], actions + [action_idx]))

    return {
        'path': [],
        'actions': [],
        'nodes_expanded': nodes_expanded,
        'time_taken': time.perf_counter() - start_time,
        'success': False,
        'cost': float('inf')
    }


def space_time_astar(env, initial_phase=0):
    """
    A* Search on the Space-Time graph (r, c, t) using Manhattan Distance Heuristic.
    
    Heuristic h(r, c, t) = manhattan_distance((r, c), target)
    Admissible and consistent since agent moves at most 1 grid cell per timestep.
    
    Returns:
        dict containing:
            'path': list of (r, c) positions
            'actions': list of actions taken
            'nodes_expanded': number of expanded search states
            'time_taken': computation time in seconds
            'success': True if target reached, False otherwise
    """
    start_time = time.perf_counter()
    start_pos = env.start
    target_pos = env.target
    max_steps = env.max_steps

    # Priority queue stores: (f_score, g_score, curr_pos, t, path, actions)
    # Counter tie-breaker for deterministic ordering in heap
    counter = 0
    start_h = manhattan_distance(start_pos, target_pos)
    open_set = [(start_h, 0, counter, start_pos, 0, [start_pos], [])]
    
    g_scores = {(start_pos, 0): 0}
    nodes_expanded = 0

    while open_set:
        f, g, _, curr_pos, t, path, actions = heapq.heappop(open_set)
        nodes_expanded += 1

        if curr_pos == target_pos:
            return {
                'path': path,
                'actions': actions,
                'nodes_expanded': nodes_expanded,
                'time_taken': time.perf_counter() - start_time,
                'success': True,
                'cost': len(actions)
            }

        if t >= max_steps:
            continue

        for action_idx, (dr, dc) in env.ACTIONS.items():
            next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            next_t = t + 1

            if not env.is_valid_cell(next_pos):
                continue

            if env.is_detected(next_pos, t=next_t + initial_phase):
                continue

            next_g = g + 1
            next_state = (next_pos, next_t)

            if next_state not in g_scores or next_g < g_scores[next_state]:
                g_scores[next_state] = next_g
                h = manhattan_distance(next_pos, target_pos)
                next_f = next_g + h
                counter += 1
                heapq.heappush(open_set, (next_f, next_g, counter, next_pos, next_t, path + [next_pos], actions + [action_idx]))

    return {
        'path': [],
        'actions': [],
        'nodes_expanded': nodes_expanded,
        'time_taken': time.perf_counter() - start_time,
        'success': False,
        'cost': float('inf')
    }
