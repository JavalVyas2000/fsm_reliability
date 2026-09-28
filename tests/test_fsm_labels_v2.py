import pytest

from src.evaluation.fsm_labels_v2 import find_first_json_object, label_answer

G = {0: [1, 2], 1: [3], 2: [3], 3: []}
SP = [0, 1, 3]


def lab(text):
    return label_answer(text, G, 0, 3, SP, num_nodes=4)


def test_valid_optimal():
    r = lab('{"path": [0, 1, 3]}')
    assert (r["schema_valid"], r["valid_path"], r["optimal_path"], r["candidate_invalid"]) == (1, 1, 1, 0)


def test_valid_suboptimal_is_not_a_failure():
    g = {0: [1, 3], 1: [3], 3: []}
    r = label_answer('{"path": [0, 1, 3]}', g, 0, 3, [0, 3], num_nodes=4)
    assert r["candidate_invalid"] == 0 and r["suboptimal_valid"] == 1


def test_invalid_edge():
    r = lab('{"path": [0, 3]}')
    assert r["schema_valid"] == 1 and r["candidate_invalid"] == 1


@pytest.mark.parametrize(
    "text,reason",
    [
        ('{"path": [0, 1', "no_complete_json_object"),
        ("no json here", "no_complete_json_object"),
        ('{"route": [0, 3]}', "missing_path_key"),
        ('{"path": ["0", "3"]}', "non_int_element"),
        ('{"path": [0, true, 3]}', "non_int_element"),
        ('{"path": [0, 9]}', "node_out_of_range"),
        ('{"path": []}', "path_length_out_of_range"),
        ('{"path": 3}', "path_not_list"),
    ],
)
def test_format_failures_are_not_semantic_labels(text, reason):
    r = lab(text)
    assert r["schema_valid"] == 0
    assert r["candidate_invalid"] is None
    assert r["format_failure_reason"] == reason


def test_first_object_only_and_braces_in_strings():
    text = '```json\n{"path": [0, 2, 3], "note": "a } b"}\n``` {"path": [0, 3]}'
    s, e = find_first_json_object(text)
    assert text[s:e] == '{"path": [0, 2, 3], "note": "a } b"}'
    assert lab(text)["valid_path"] == 1
