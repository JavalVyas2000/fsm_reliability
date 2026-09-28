"""
FSM prompt v2 (protocol v1.0, section 4).

Changes from v1 (fsm_prompts.build_path_prompt):
  - chat format with a fixed system message (rendered by the tokenizer's template);
  - no empty-path option (every query is reachable by construction);
  - the few-shot example uses a fixed 6-node graph that never occurs in any split,
    with a 4-node answer, so it cannot be copied into an instance answer pattern;
  - the user message is assembled from named segments so attention regions can be
    located by character offsets rather than by searching literal strings.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.utils.manifest import sha256_text

PROMPT_VERSION = "fsm_v2.0"

SYSTEM_MSG = "You are a precise graph-search assistant. You answer with a single JSON object only."

EXAMPLE_GRAPH: Dict[int, List[int]] = {
    0: [3, 5],
    1: [2],
    2: [0],
    3: [1, 4],
    4: [2],
    5: [4],
}
EXAMPLE_START, EXAMPLE_GOAL = 0, 2
EXAMPLE_ANSWER = [0, 3, 4, 2]

_INSTRUCTION = (
    "A directed graph is given as an adjacency list. Each line `u: [v1, v2, ...]` "
    "means there is a directed edge from state u to each listed state v.\n\n"
    "Find a path from the start state to the goal state that follows directed edges. "
    "Prefer a shortest path. The path is a list of states that begins with the start "
    "state and ends with the goal state. A path always exists.\n\n"
    'Respond with exactly one JSON object of the form {"path": [s0, s1, ..., sk]} '
    "and nothing else.\n\n"
)


def _graph_text(graph: Dict[int, List[int]]) -> str:
    return "\n".join(
        f"{u}: [{', '.join(str(v) for v in graph[u])}]" for u in sorted(graph)
    )


def _example_block() -> str:
    return (
        "Example\nGraph:\n"
        f"{_graph_text(EXAMPLE_GRAPH)}\n"
        f"Start state: {EXAMPLE_START}\nGoal state: {EXAMPLE_GOAL}\n"
        f'Answer: {{"path": [{", ".join(str(x) for x in EXAMPLE_ANSWER)}]}}\n\n'
    )


def build_user_segments(graph_text: str, start: int, goal: int) -> List[Tuple[str, str]]:
    """Ordered (region_name, text) segments whose concatenation is the user message."""
    return [
        ("instruction", _INSTRUCTION),
        ("example", _example_block()),
        ("instruction", "Now solve this instance.\nGraph:\n"),
        ("graph", graph_text),
        ("instruction", "\n"),
        ("query", f"Start state: {start}\nGoal state: {goal}"),
    ]


def build_messages(graph_text: str, start: int, goal: int) -> List[Dict[str, str]]:
    user = "".join(t for _, t in build_user_segments(graph_text, start, goal))
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user},
    ]


def region_char_spans(
    rendered: str, graph_text: str, start: int, goal: int
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Character spans of each region inside the chat-template-rendered prompt.
    Characters not covered by any region belong to the template (special tokens,
    role headers).
    """
    spans: Dict[str, List[Tuple[int, int]]] = {}
    sys_at = rendered.find(SYSTEM_MSG)
    if sys_at < 0:
        raise ValueError("System message not found in rendered prompt")
    spans["system"] = [(sys_at, sys_at + len(SYSTEM_MSG))]

    segments = build_user_segments(graph_text, start, goal)
    user = "".join(t for _, t in segments)
    user_at = rendered.find(user)
    if user_at < 0 or rendered.find(user, user_at + 1) >= 0:
        raise ValueError("User message not found exactly once in rendered prompt")
    pos = user_at
    for name, text in segments:
        spans.setdefault(name, []).append((pos, pos + len(text)))
        pos += len(text)
    return spans


def prompt_template_hash() -> str:
    """Hash of every fixed prompt component (instance fields replaced by placeholders)."""
    parts = [PROMPT_VERSION, SYSTEM_MSG]
    parts += [t for _, t in build_user_segments("{GRAPH}", -1, -2)]
    return sha256_text("\x1f".join(parts))
