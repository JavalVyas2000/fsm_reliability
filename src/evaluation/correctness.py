from __future__ import annotations

from typing import Optional

from src.data.labels import validate_path_label


def score_prediction(
    graph: dict[int, list[int]],
    predicted_path: Optional[list[int]],
    start: int,
    goal: int,
    shortest_path: list[int],
) -> dict:
    """
    Score a parsed model prediction.

    If parsing fails, treat it as invalid.
    """
    if predicted_path is None:
        return {
            "parsed_path": None,
            "valid_path": 0,
            "optimal_path": 0,
            "path_length": 0,
            "shortest_length": len(shortest_path) if shortest_path else 0,
            "parse_success": 0,
        }

    labels = validate_path_label(
        graph=graph,
        candidate_path=predicted_path,
        start=start,
        goal=goal,
        shortest_path=shortest_path,
    )
    labels["parsed_path"] = predicted_path
    labels["parse_success"] = 1
    return labels