"""
Strict answer parsing and labels for FSM prompt v2 (protocol v1.0, section 2).

parse_success : the first balanced JSON object parses and has key "path"
schema_valid  : path is a list of ints (no bools/strings) in [0, num_nodes),
                length 1..4*num_nodes
candidate_invalid : schema_valid and not valid_path   (routing target, positive = failure)
suboptimal_valid  : valid_path and not optimal_path   (secondary; not a routing failure)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.data.labels import is_optimal_path, is_valid_path


def find_first_json_object(text: str) -> Optional[Tuple[int, int]]:
    """Char span [start, end) of the first brace-balanced object, respecting JSON strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def parse_answer(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "parse_success": 0,
        "parsed_path": None,
        "json_span": None,
        "format_failure_reason": None,
    }
    span = find_first_json_object(text)
    if span is None:
        out["format_failure_reason"] = "no_complete_json_object"
        return out
    out["json_span"] = list(span)
    try:
        payload = json.loads(text[span[0] : span[1]])
    except json.JSONDecodeError:
        out["format_failure_reason"] = "json_decode_error"
        return out
    if not isinstance(payload, dict) or "path" not in payload:
        out["format_failure_reason"] = "missing_path_key"
        return out
    out["parse_success"] = 1
    out["parsed_path"] = payload["path"]
    return out


def schema_check(path: Any, num_nodes: int) -> Tuple[bool, Optional[str]]:
    if not isinstance(path, list):
        return False, "path_not_list"
    if not (1 <= len(path) <= 4 * num_nodes):
        return False, "path_length_out_of_range"
    for x in path:
        if isinstance(x, bool) or not isinstance(x, int):
            return False, "non_int_element"
        if not (0 <= x < num_nodes):
            return False, "node_out_of_range"
    return True, None


def label_answer(
    text: str,
    graph: Dict[int, List[int]],
    start: int,
    goal: int,
    shortest_path: List[int],
    num_nodes: int,
) -> Dict[str, Any]:
    rec = parse_answer(text)
    rec.update(
        schema_valid=0,
        valid_path=0,
        optimal_path=0,
        candidate_invalid=None,
        suboptimal_valid=None,
        label_status="known",
    )
    if not rec["parse_success"]:
        return rec
    ok, reason = schema_check(rec["parsed_path"], num_nodes)
    if not ok:
        rec["format_failure_reason"] = reason
        return rec
    path = rec["parsed_path"]
    valid = is_valid_path(graph, path, start, goal)
    optimal = is_optimal_path(graph, path, start, goal, shortest_path)
    rec.update(
        schema_valid=1,
        valid_path=int(valid),
        optimal_path=int(optimal),
        candidate_invalid=int(not valid),
        suboptimal_valid=int(valid and not optimal),
    )
    return rec
