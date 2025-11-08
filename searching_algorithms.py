from utils import *
from collections import deque, defaultdict
from queue import PriorityQueue
from grid import Grid
from spot import Spot
from math import sqrt
import math
import time

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    if start is None or end is None:
        return False

    ueue = deque()
    ueue.append(start)
    visited = {start}
    came_from = dict()

    while len(ueue) > 0:
        draw()
        current = ueue.popleft()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                ueue.append(neighbor)
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    stack = [start]
    visited = {start}
    came_from = dict()
    while len(stack) > 0:
        draw()
        current = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    depth_limit = 10

    stack = [start]
    depth = dict()
    depth[start] = 0
    stopped_at = deque()
    came_from = dict()
    past = set()
    current_era = set()
    while True:
        if len(stack) == 0 and len(stopped_at) != 0:
            next_current = stopped_at.popleft()
            depth[next_current] = 0
            stack.append(next_current)
            past.update(current_era)
        elif len(stack) == 0 and len(stopped_at) == 0:
            break

        draw()
        current = stack.pop()
        current_era.add(current)
        if depth[current] > depth_limit:
            stopped_at.append(current)
            continue

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in past and not neighbor.is_barrier() and depth.get(neighbor, math.inf) > depth[current] + 1:
                came_from[neighbor] = current
                depth[neighbor] = depth[current] + 1
                stack.append(neighbor)
                neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Euclidian distance between p1 and p2.
    """
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    heuristic = h_euclidian_distance
    # heuristic = h_manhattan_distance

    count = 0

    open_heap = PriorityQueue()
    open_heap.put((0, count, start))

    came_from = dict()

    g_score = dict()
    g_score[start] = 0

    f_score = dict()
    f_score[start] = heuristic(start.get_position(), end.get_position())

    lookup = set([start])

    while not open_heap.empty():
        draw()
        f, c, current = open_heap.get()
        lookup.remove(current)

        if current.get_position() == end.get_position():
            current = came_from[current]
            while current.get_position() != start.get_position():
                current.make_path()
                current = came_from[current]
                draw()
            return True

        for n in current.neighbors:
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(n, math.inf):
                came_from[n] = current
                g_score[n] = tentative_g
                f_score[n] = tentative_g + heuristic(n.get_position(), end.get_position())
                if n not in lookup:
                    count += 1
                    open_heap.put((f_score[n], count, n))
                    lookup.add(n)
                    n.make_open()

        if current.get_position() != start.get_position():
            current.make_closed()

    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    open_heap = PriorityQueue()
    open_heap.put((0, start))

    came_from = dict()

    g_score = dict()
    g_score[start] = 0

    visited = set([start])

    while not open_heap.empty():
        draw()
        f, current = open_heap.get()

        if current.get_position() == end.get_position():
            current = came_from[current]
            while current.get_position() != start.get_position():
                current.make_path()
                current = came_from[current]
                draw()
            return True

        for n in current.neighbors:
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(n, math.inf):
                came_from[n] = current
                g_score[n] = tentative_g
                if n not in visited:
                    open_heap.put((g_score[n], n))
                    visited.add(n)
                    n.make_open()

        draw()

        if current.get_position() != start.get_position():
            current.make_closed()

    return False

def dijkstra(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    open_heap = PriorityQueue()
    open_heap.put((0, start))

    came_from = dict()

    g_score = dict()
    g_score[start] = 0

    visited = set([start])

    while not open_heap.empty():
        draw()
        f, current = open_heap.get()

        if current.get_position() == end.get_position():
            current = came_from[current]
            while current.get_position() != start.get_position():
                current.make_path()
                current = came_from[current]
                draw()
            return True

        for n in current.neighbors:
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(n, math.inf):
                came_from[n] = current
                g_score[n] = tentative_g
                if n not in visited:
                    open_heap.put((g_score[n], n))
                    visited.add(n)
                    n.make_open()

        draw()

        if current.get_position() != start.get_position():
            current.make_closed()

    return False

def iddfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    came_from = dict()
    def dls_helper(depth_limit: int) -> (bool, bool):
        stack = [start]
        depth = dict()
        depth[start] = 0
        more = False
        while len(stack) > 0:
            draw()
            current = stack.pop()
            if depth[current] > depth_limit:
                more = True
                continue
            if current == end:
                return (True, False)

            if current != start:
                current.make_closed()

            for neighbor in current.neighbors:
                if not neighbor.is_barrier() and depth.get(neighbor, math.inf) > depth[current] + 1:
                        came_from[neighbor] = current
                        depth[neighbor] = depth[current] + 1
                        stack.append(neighbor)
                        neighbor.make_open()

        return (False, more)

    depth = 0
    more = True
    while more:
        for row in grid.grid:
            for cell in row:
                if cell.is_closed() or cell.is_open():
                    cell.reset()
        found, more = dls_helper(depth)
        if found:
            current = end
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        depth += 1
    return False

def ida(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    # heuristic = h_euclidian_distance
    heuristic = h_manhattan_distance

    def dfs_helper() -> (dict, int):
        stack = [(0, start)]
        came_from = dict()
        visited = set()
        pruned_bound = math.inf

        while len(stack) > 0:
            draw()
            cost, current = stack.pop()
            if current.get_position() == end.get_position():
                return (came_from, 0)

            if current.get_position() != start.get_position():
                current.make_closed()

            for n in current.neighbors:
                if n not in visited:
                    came_from[n] = current
                    visited.add(n)
                    new_cost = cost + 1
                    new_bound = heuristic(n.get_position(), end.get_position()) + new_cost
                    if new_bound > bound:
                        if new_bound < pruned_bound:
                            pruned_bound = new_bound
                    else:
                        stack.append((new_cost, n))
                        n.make_open()
        return (None, pruned_bound)


    bound = heuristic(start.get_position(), end.get_position())
    print(bound)
    found = False
    while not found:
        for row in grid.grid:
            for cell in row:
                if cell.is_closed() or cell.is_open():
                    cell.reset()
        came_from, new_bound = dfs_helper()
        draw()
        if came_from is not None:
            current = end
            while current.get_position() != start.get_position():
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        bound = new_bound

    return False

