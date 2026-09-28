"""
Generate a graph-disjoint FSM dataset (protocol v1.0, section 5.1).

Example:
    python -m scripts.10_generate_fsm_v2 --root_seed 20260923 --tag fsm_pilot \
        --sizes train=1500 dev_cal=500 dev_thr=500 test_iid=500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.fsm_dataset_v2 import (
    DEFAULT_EDGE_PROB,
    GENERATOR_VERSION,
    generate_partitions,
    graph_hash,
    graph_hashes_from_csvs,
    to_dataframe,
)
from src.prompts.fsm_prompts_v2 import EXAMPLE_GRAPH
from src.utils.manifest import REPO_ROOT, build_manifest, sha256_file, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_seed", type=int, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--sizes", nargs="+", required=True, help="partition=count ...")
    p.add_argument("--num_nodes", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--out_root", type=str, default="data/v2")
    p.add_argument(
        "--exclude_glob",
        type=str,
        nargs="+",
        default=["data/raw*/*.csv", "data/v2/*/*.csv"],
        help="existing datasets whose graphs must never reappear",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sizes = {k: int(v) for k, v in (s.split("=") for s in args.sizes)}
    out_dir = REPO_ROOT / args.out_root / f"{args.tag}_seed{args.root_seed}"
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset: {out_dir}")

    excluded_files = sorted({p for g in args.exclude_glob for p in REPO_ROOT.glob(g)})
    excluded = graph_hashes_from_csvs(excluded_files)
    excluded.add(graph_hash(EXAMPLE_GRAPH))

    edge_prob = {n: DEFAULT_EDGE_PROB[n] for n in args.num_nodes}
    parts, stats = generate_partitions(
        root_seed=args.root_seed,
        sizes=sizes,
        num_nodes_list=args.num_nodes,
        edge_prob=edge_prob,
        excluded_graph_hashes=excluded,
    )

    out_dir.mkdir(parents=True)
    files = {}
    for name, instances in parts.items():
        path = out_dir / f"{name}.csv"
        to_dataframe(instances).to_csv(path, index=False)
        files[name] = {"path": str(path.relative_to(REPO_ROOT)), "rows": len(instances), "sha256": sha256_file(path)}

    all_hashes = [x.graph_hash for inst in parts.values() for x in inst]
    checks = {
        "graph_disjoint_across_partitions": len(all_hashes) == len(set(all_hashes)),
        "overlap_with_excluded": len(set(all_hashes) & excluded),
        "excluded_graph_count": len(excluded),
        "excluded_files": [str(p.relative_to(REPO_ROOT)) for p in excluded_files],
    }
    if not checks["graph_disjoint_across_partitions"] or checks["overlap_with_excluded"]:
        raise SystemExit(f"Disjointness check failed: {checks}")

    manifest = build_manifest(
        dataset={
            "generator_version": GENERATOR_VERSION,
            "root_seed": args.root_seed,
            "sizes": sizes,
            "num_nodes": args.num_nodes,
            "edge_prob": edge_prob,
            "files": files,
            "generation_stats": stats,
            "checks": checks,
        }
    )
    write_json(out_dir / "dataset_manifest.json", manifest)
    print(json.dumps({"out_dir": str(out_dir), "files": files, "checks": {k: v for k, v in checks.items() if k != "excluded_files"}}, indent=2))


if __name__ == "__main__":
    main()
