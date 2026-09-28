"""
Observable task-context and proposed-action features (protocol v1.0, section 6).
These are not model internals; they are the baseline that internals must beat.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

FSM_CONTEXT_ACTION = [
    "num_nodes",
    "num_edges",
    "edge_density",
    "shortest_length",
    "answer_length",
    "answer_is_two_node",
    "answer_repeats_node",
    "answer_endpoints_match",
]
FSM_CONTEXT_ACTION_NO_BFS = [c for c in FSM_CONTEXT_ACTION if c != "shortest_length"]


def fsm_context_action(
    num_nodes: int,
    num_edges: int,
    shortest_length: int,
    start: int,
    goal: int,
    path: Optional[List[Any]],
) -> Dict[str, float]:
    feats: Dict[str, float] = {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "edge_density": num_edges / float(num_nodes * (num_nodes - 1)),
        "shortest_length": float(shortest_length),
    }
    if not isinstance(path, list) or not path:
        feats.update(
            answer_length=np.nan,
            answer_is_two_node=np.nan,
            answer_repeats_node=np.nan,
            answer_endpoints_match=np.nan,
        )
        return feats
    feats.update(
        answer_length=float(len(path)),
        answer_is_two_node=float(len(path) == 2),
        answer_repeats_node=float(len(set(map(str, path))) < len(path)),
        answer_endpoints_match=float(path[0] == start and path[-1] == goal),
    )
    return feats
