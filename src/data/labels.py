from __future__ import annotations

from typing import Dict, List, Tuple

AdjacencyDict = Dict[int, List[int]]


def is_valid_path(
    graph: AdjacencyDict,
    path: List[int],
    start: int,
    goal: int,
) -> bool:
    """
    Check whether a path:
    1. is non-empty
    2. starts at start
    3. ends at goal
    4. uses only valid directed edges
    """
    if not path:
        return False

    if path[0] != start:
        return False

    if path[-1] != goal:
        return False

    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]

        if src not in graph:
            return False
        if dst not in graph[src]:
            return False

    return True


def is_optimal_path(
    candidate_path: List[int],
    shortest_path: List[int],
) -> bool:
    """
    Check whether the candidate path has the same length as the shortest path.
    """
    if not candidate_path or not shortest_path:
        return False
    return len(candidate_path) == len(shortest_path)


def path_to_actions(path: List[int]) -> List[Tuple[int, int]]:
    """
    Convert a node path into edge actions.

    Example:
        [0, 2, 5] -> [(0, 2), (2, 5)]
    """
    if len(path) < 2:
        return []
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def validate_path_label(
    graph: AdjacencyDict,
    candidate_path: List[int],
    start: int,
    goal: int,
    shortest_path: List[int],
) -> dict:
    """
    Return a dictionary of labels for the candidate path.
    """
    valid = is_valid_path(graph, candidate_path, start, goal)
    optimal = valid and is_optimal_path(candidate_path, shortest_path)

    return {
        "valid_path": int(valid),
        "optimal_path": int(optimal),
        "path_length": len(candidate_path) if candidate_path else 0,
        "shortest_length": len(shortest_path) if shortest_path else 0,
    }