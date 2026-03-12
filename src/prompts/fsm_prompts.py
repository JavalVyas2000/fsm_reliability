from __future__ import annotations


def build_path_prompt(graph_text: str, start: int, goal: int) -> str:
    """
    Fixed ETFA prompt template with JSON-only output.
    """
    return f"""You are given a directed graph as an adjacency list.

Graph:
{graph_text}

Task:
Find one valid path from state {start} to state {goal}.

Requirements:
- Return exactly one JSON object.
- Do not return any explanation.
- Do not return markdown.
- Do not use code fences.
- The JSON must have exactly one key: "path".
- The value of "path" must be a list of integers.
- The path must start with {start}.
- The path must end with {goal}.
- Each consecutive pair of states must be a valid directed edge in the graph.
- If no valid path exists, return {{"path": []}}.

Valid output examples:
{{"path": [{start}, {goal}]}}
{{"path": [{start}, 3, {goal}]}}
{{"path": []}}

Output:
""".strip()


def build_path_prompt_chat(graph_text: str, start: int, goal: int) -> list[dict]:
    """
    Chat-style prompt for instruct/chat models.
    """
    system_msg = (
        "You are a precise graph traversal assistant. "
        "Return exactly one JSON object and nothing else."
    )

    user_msg = f"""You are given a directed graph as an adjacency list.

Graph:
{graph_text}

Find one valid path from state {start} to state {goal}.

Requirements:
- Return exactly one JSON object.
- No explanation.
- No markdown.
- No code fences.
- The JSON must have exactly one key: "path".
- The value of "path" must be a list of integers.
- The path must start with {start}.
- The path must end with {goal}.
- Each consecutive pair of states must be a valid directed edge.
- If no valid path exists, return {{"path": []}}.
""".strip()

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]