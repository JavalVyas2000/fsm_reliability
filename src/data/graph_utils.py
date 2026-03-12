from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple


AdjacencyDict = Dict[int, List[int]]


def generate_directed_graph(
    num_nodes: int,
    edge_prob: float,
    seed: Optional[int] = None,
    allow_self_loops: bool = False,
) -> AdjacencyDict:
    """
    Generate a random directed graph as an adjacency dictionary.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_prob: Probability of creating a directed edge i -> j.
        seed: Optional RNG seed for reproducibility.
        allow_self_loops: Whether to allow i -> i edges.

    Returns:
        Dictionary mapping node -> sorted list of outbound neighbors.
    """
    if num_nodes <= 1:
        raise ValueError("num_nodes must be > 1")
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be in [0, 1]")

    rng = random.Random(seed)
    graph: AdjacencyDict = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        for j in range(num_nodes):
            if not allow_self_loops and i == j:
                continue
            if rng.random() < edge_prob:
                graph[i].append(j)

    for node in graph:
        graph[node] = sorted(set(graph[node]))

    return graph


def count_edges(graph: AdjacencyDict) -> int:
    """Return the total number of directed edges."""
    return sum(len(neighbors) for neighbors in graph.values())


def shortest_path(
    graph: AdjacencyDict,
    start: int,
    goal: int,
) -> Optional[List[int]]:
    """
    Compute the shortest path from start to goal using BFS.

    Args:
        graph: Adjacency dictionary.
        start: Source node.
        goal: Destination node.

    Returns:
        List of nodes in the shortest path, or None if unreachable.
    """
    if start not in graph or goal not in graph:
        raise ValueError("start and goal must both exist in graph")

    if start == goal:
        return [start]

    visited = set([start])
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()

        for neighbor in graph[current]:
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == goal:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def path_exists(graph: AdjacencyDict, start: int, goal: int) -> bool:
    """Return True if a path exists from start to goal."""
    return shortest_path(graph, start, goal) is not None


def adjacency_dict_to_string(graph: AdjacencyDict) -> str:
    """
    Serialize adjacency dict into a compact stable string for prompting/logging.

    Example:
        0: [1, 3]
        1: [2]
        2: []
    """
    lines = []
    for node in sorted(graph.keys()):
        nbrs = ", ".join(str(n) for n in graph[node])
        lines.append(f"{node}: [{nbrs}]")
    return "\n".join(lines)


def sample_reachable_start_goal(
    graph: AdjacencyDict,
    rng: random.Random,
    max_tries: int = 200,
) -> Optional[Tuple[int, int, List[int]]]:
    """
    Sample a (start, goal) pair with at least one reachable path.

    Returns:
        (start, goal, shortest_path) or None if no such pair is found.
    """
    nodes = list(graph.keys())
    if len(nodes) < 2:
        return None

    for _ in range(max_tries):
        start, goal = rng.sample(nodes, 2)
        sp = shortest_path(graph, start, goal)
        if sp is not None:
            return start, goal, sp

    return None