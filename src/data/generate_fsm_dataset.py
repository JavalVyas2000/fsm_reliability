from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .graph_utils import (
    adjacency_dict_to_string,
    count_edges,
    generate_directed_graph,
    sample_reachable_start_goal,
)


@dataclass
class FSMInstance:
    instance_id: str
    split: str
    num_nodes: int
    edge_prob: float
    num_edges: int
    start: int
    goal: int
    shortest_path: List[int]
    shortest_path_length: int
    graph_json: str
    graph_text: str


def build_instances(
    split: str,
    num_samples: int,
    num_nodes: int,
    edge_prob: float,
    seed: int,
) -> List[FSMInstance]:
    """
    Build a list of reachable FSM traversal instances.
    """
    rng = random.Random(seed)
    instances: List[FSMInstance] = []

    trials = 0
    while len(instances) < num_samples:
        graph_seed = rng.randint(0, 10_000_000)
        graph = generate_directed_graph(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            seed=graph_seed,
            allow_self_loops=False,
        )

        sampled = sample_reachable_start_goal(graph, rng)
        trials += 1

        if sampled is None:
            if trials > num_samples * 50:
                raise RuntimeError(
                    "Could not sample enough reachable instances. "
                    "Try increasing edge_prob or reducing num_nodes."
                )
            continue

        start, goal, sp = sampled

        instance = FSMInstance(
            instance_id=f"{split}_n{num_nodes}_p{str(edge_prob).replace('.', '')}_{len(instances):05d}",
            split=split,
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            num_edges=count_edges(graph),
            start=start,
            goal=goal,
            shortest_path=sp,
            shortest_path_length=len(sp),
            graph_json=json.dumps(graph, sort_keys=True),
            graph_text=adjacency_dict_to_string(graph),
        )
        instances.append(instance)

    return instances


def instances_to_dataframe(instances: List[FSMInstance]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in instances])


def save_split(
    output_path: Path,
    instances: List[FSMInstance],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = instances_to_dataframe(instances)
    df.to_csv(output_path, index=False)


def build_default_dataset(
    output_dir: str = "data/raw",
    train_samples: int = 1000,
    val_samples: int = 200,
    test_samples: int = 200,
    num_nodes: int = 10,
    edge_prob: float = 0.30,
    seed: int = 42,
) -> None:
    """
    Build default train/val/test splits.
    """
    output_root = Path(output_dir)

    train = build_instances(
        split="train",
        num_samples=train_samples,
        num_nodes=num_nodes,
        edge_prob=edge_prob,
        seed=seed,
    )
    val = build_instances(
        split="val",
        num_samples=val_samples,
        num_nodes=num_nodes,
        edge_prob=edge_prob,
        seed=seed + 1,
    )
    test = build_instances(
        split="test",
        num_samples=test_samples,
        num_nodes=num_nodes,
        edge_prob=edge_prob,
        seed=seed + 2,
    )

    save_split(output_root / "train.csv", train)
    save_split(output_root / "val.csv", val)
    save_split(output_root / "test.csv", test)

    print(f"Saved dataset to: {output_root.resolve()}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")