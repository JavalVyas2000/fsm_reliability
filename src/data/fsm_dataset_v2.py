"""
Graph-disjoint FSM traversal dataset generator (protocol v1.0, section 5.1).

Differences from generate_fsm_dataset.py:
  - independent seed streams per partition and node count (numpy SeedSequence),
    so no two partitions or root seeds share an RNG stream;
  - every graph appears at most once across all partitions (graph-disjoint),
    and never matches an excluded graph (existing raw data, prompt example);
  - content-hash ids and group keys, dataset fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd

from .graph_utils import (
    AdjacencyDict,
    adjacency_dict_to_string,
    count_edges,
    generate_directed_graph,
    sample_reachable_start_goal,
)

GENERATOR_VERSION = "fsm_dataset_v2.0"

DEFAULT_EDGE_PROB: Dict[int, float] = {5: 0.35, 10: 0.30, 15: 0.25, 20: 0.20}
PARTITIONS = ("train", "dev_cal", "dev_thr", "cert", "test_iid")


def canonical_graph(graph: Mapping) -> List[List]:
    """Canonical form: sorted [node, sorted(neighbours)] with int nodes."""
    return sorted([int(k), sorted(int(v) for v in vs)] for k, vs in graph.items())


def graph_hash(graph: Mapping) -> str:
    payload = json.dumps(canonical_graph(graph), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_hash(graph: Mapping, start: int, goal: int) -> str:
    payload = json.dumps(
        [canonical_graph(graph), int(start), int(goal)], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_hashes_from_csvs(paths: Iterable[Path]) -> Set[str]:
    """Graph hashes of every row in existing dataset CSVs (column graph_json)."""
    hashes: Set[str] = set()
    for p in paths:
        df = pd.read_csv(p, usecols=["graph_json"])
        for gj in df["graph_json"]:
            hashes.add(graph_hash(json.loads(gj)))
    return hashes


@dataclass
class FSMInstanceV2:
    instance_id: str
    partition: str
    num_nodes: int
    edge_prob: float
    num_edges: int
    start: int
    goal: int
    shortest_path: str  # JSON list
    shortest_path_length: int
    graph_json: str
    graph_text: str
    graph_hash: str
    query_hash: str


def _allocate(total: int, keys: Sequence[int]) -> Dict[int, int]:
    base, rem = divmod(total, len(keys))
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def generate_partitions(
    root_seed: int,
    sizes: Mapping[str, int],
    num_nodes_list: Sequence[int] = (5, 10, 15, 20),
    edge_prob: Mapping[int, float] = DEFAULT_EDGE_PROB,
    excluded_graph_hashes: Set[str] = frozenset(),
    max_attempts_factor: int = 200,
) -> tuple[Dict[str, List[FSMInstanceV2]], Dict[str, int]]:
    """
    Generate graph-disjoint partitions.

    Partition p (in PARTITIONS order) uses SeedSequence(root_seed).spawn(len(PARTITIONS))[i],
    further spawned per node count. One query per graph, so the graph is the
    independence unit. Returns (partitions, stats).
    """
    unknown = set(sizes) - set(PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown partitions: {sorted(unknown)}")

    seen: Set[str] = set(excluded_graph_hashes)
    children = np.random.SeedSequence(root_seed).spawn(len(PARTITIONS))
    out: Dict[str, List[FSMInstanceV2]] = {}
    stats = {"rejected_duplicate_or_excluded": 0, "rejected_no_reachable_pair": 0}

    for part_idx, part in enumerate(PARTITIONS):
        n_total = int(sizes.get(part, 0))
        if n_total <= 0:
            continue
        alloc = _allocate(n_total, list(num_nodes_list))
        node_seqs = children[part_idx].spawn(len(num_nodes_list))
        instances: List[FSMInstanceV2] = []

        for n_idx, num_nodes in enumerate(num_nodes_list):
            rng = np.random.default_rng(node_seqs[n_idx])
            ep = float(edge_prob[num_nodes])
            made, attempts = 0, 0
            while made < alloc[num_nodes]:
                attempts += 1
                if attempts > max_attempts_factor * max(1, alloc[num_nodes]):
                    raise RuntimeError(
                        f"Could not generate enough unique graphs for {part}, n={num_nodes}"
                    )
                graph: AdjacencyDict = generate_directed_graph(
                    num_nodes=num_nodes,
                    edge_prob=ep,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    allow_self_loops=False,
                )
                gh = graph_hash(graph)
                if gh in seen:
                    stats["rejected_duplicate_or_excluded"] += 1
                    continue
                pair_rng = random.Random(int(rng.integers(0, 2**31 - 1)))
                sampled = sample_reachable_start_goal(graph, pair_rng)
                if sampled is None:
                    stats["rejected_no_reachable_pair"] += 1
                    continue
                seen.add(gh)
                start, goal, sp = sampled
                instances.append(
                    FSMInstanceV2(
                        instance_id=f"v2_{part}_{gh[:16]}",
                        partition=part,
                        num_nodes=num_nodes,
                        edge_prob=ep,
                        num_edges=count_edges(graph),
                        start=int(start),
                        goal=int(goal),
                        shortest_path=json.dumps([int(x) for x in sp]),
                        shortest_path_length=len(sp),
                        graph_json=json.dumps(graph, sort_keys=True),
                        graph_text=adjacency_dict_to_string(graph),
                        graph_hash=gh,
                        query_hash=query_hash(graph, start, goal),
                    )
                )
                made += 1

        # Deterministic shuffle so node counts are interleaved.
        order = np.random.default_rng(children[part_idx].spawn(1)[0]).permutation(len(instances))
        out[part] = [instances[i] for i in order]

    return out, stats


def to_dataframe(instances: List[FSMInstanceV2]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in instances])


def load_graph(row: Mapping) -> AdjacencyDict:
    return {int(k): [int(v) for v in vs] for k, vs in json.loads(row["graph_json"]).items()}
