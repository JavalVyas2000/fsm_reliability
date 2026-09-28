from src.data.fsm_dataset_v2 import graph_hash
from src.data.graph_utils import shortest_path
from src.prompts.fsm_prompts_v2 import (
    EXAMPLE_ANSWER,
    EXAMPLE_GOAL,
    EXAMPLE_GRAPH,
    EXAMPLE_START,
    build_messages,
    prompt_template_hash,
    region_char_spans,
)

GRAPH_TEXT = "0: [1]\n1: [2]\n2: []"


def fake_render(messages):
    # Mimics a chat template: role headers and special tokens around each message.
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def test_example_answer_is_a_shortest_path_and_not_an_instance_pattern():
    sp = shortest_path(EXAMPLE_GRAPH, EXAMPLE_START, EXAMPLE_GOAL)
    assert len(sp) == len(EXAMPLE_ANSWER) == 4
    # every consecutive pair is an edge
    assert all(b in EXAMPLE_GRAPH[a] for a, b in zip(EXAMPLE_ANSWER, EXAMPLE_ANSWER[1:]))
    assert len(EXAMPLE_GRAPH) not in (5, 10, 15, 20)


def test_no_empty_path_option():
    user = build_messages(GRAPH_TEXT, 0, 2)[1]["content"]
    assert '"path": []' not in user


def test_region_spans_recover_exact_text():
    rendered = fake_render(build_messages(GRAPH_TEXT, 0, 2))
    spans = region_char_spans(rendered, GRAPH_TEXT, 0, 2)
    assert [rendered[s:e] for s, e in spans["graph"]] == [GRAPH_TEXT]
    assert rendered[slice(*spans["query"][0])] == "Start state: 0\nGoal state: 2"
    assert "Example" in rendered[slice(*spans["example"][0])]
    covered = sum(e - s for v in spans.values() for s, e in v)
    uncovered = rendered
    assert covered < len(uncovered)  # template chars remain uncovered


def test_template_hash_is_instance_independent():
    assert prompt_template_hash() == prompt_template_hash()
    assert len(prompt_template_hash()) == 64
    assert graph_hash(EXAMPLE_GRAPH)
