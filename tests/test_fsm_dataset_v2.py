import json

import pandas as pd

from src.data.fsm_dataset_v2 import (
    generate_partitions,
    graph_hash,
    load_graph,
)
from src.data.graph_utils import shortest_path

SIZES = {"train": 40, "dev_cal": 12, "dev_thr": 12, "test_iid": 12}


def _hashes(parts):
    return [x.graph_hash for inst in parts.values() for x in inst]


def test_deterministic_for_same_root():
    a, _ = generate_partitions(123, SIZES)
    b, _ = generate_partitions(123, SIZES)
    assert _hashes(a) == _hashes(b)


def test_different_roots_differ():
    a, _ = generate_partitions(123, SIZES)
    b, _ = generate_partitions(124, SIZES)
    assert set(_hashes(a)) != set(_hashes(b))


def test_graph_disjoint_and_sizes():
    parts, _ = generate_partitions(7, SIZES)
    hashes = _hashes(parts)
    assert len(hashes) == len(set(hashes))
    for name, n in SIZES.items():
        assert len(parts[name]) == n
        counts = pd.Series([x.num_nodes for x in parts[name]]).value_counts()
        assert counts.max() - counts.min() <= 1


def test_excluded_graphs_never_generated():
    first, _ = generate_partitions(7, SIZES)
    excluded = set(_hashes(first))
    again, stats = generate_partitions(7, SIZES, excluded_graph_hashes=excluded)
    assert not (set(_hashes(again)) & excluded)
    assert stats["rejected_duplicate_or_excluded"] > 0


def test_instances_are_consistent():
    parts, _ = generate_partitions(11, {"train": 20})
    for x in parts["train"]:
        row = {"graph_json": x.graph_json}
        g = load_graph(row)
        assert graph_hash(g) == x.graph_hash
        sp = shortest_path(g, x.start, x.goal)
        assert sp == json.loads(x.shortest_path)
        assert len(sp) >= 2 and x.start != x.goal


def test_graph_hash_ignores_key_types_and_order():
    g1 = {0: [2, 1], 1: [], 2: [0]}
    g2 = {"2": [0], "0": [1, 2], "1": []}
    assert graph_hash(g1) == graph_hash(g2)
