"""
Map the spreadsheet splits of a scripts/35 build onto the analysis partitions
(docs/cstr_v4_prereg.md):

    train -> train          val -> dev_cal + dev_thr          test -> test_iid + cert

Within each split and family, episodes are taken in slot order and alternate between the two
halves (1st -> first half, 2nd -> second half, ...), so each half has the same family counts.

Writes snapshots_<partition>.pkl and partitions.csv next to the build, and adds the partition
sizes, file hashes and the pre-registration hash to dataset_manifest.json (read by scripts/25).

Example:
    python -m scripts.36_cstr_assign_partitions --build_dir data/cstr/episodes_v4
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter

import pandas as pd

from src.utils.manifest import REPO_ROOT, sha256_file, write_json

HALVES = {"val": ("dev_cal", "dev_thr"), "test": ("test_iid", "cert")}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build_dir", type=str, required=True)
    p.add_argument("--prereg", type=str, default="docs/cstr_v4_prereg.md")
    return p.parse_args()


def main():
    args = parse_args()
    d = REPO_ROOT / args.build_dir
    rows, parts = [], {}
    for split in ("train", "val", "test"):
        snaps = pickle.load(open(d / f"snapshots_{split}.pkl", "rb"))
        if split == "train":
            parts["train"] = snaps
            rows += [{"episode_id": s.spec.episode_id, "split": split, "family": s.spec.family, "partition": "train"} for s in snaps]
            continue
        a, b = HALVES[split]
        parts[a], parts[b] = [], []
        seen = Counter()
        for s in snaps:  # spreadsheet slot order
            target = a if seen[s.spec.family] % 2 == 0 else b
            seen[s.spec.family] += 1
            parts[target].append(s)
            rows.append({"episode_id": s.spec.episode_id, "split": split, "family": s.spec.family, "partition": target})
    for name, snaps in parts.items():
        with open(d / f"snapshots_{name}.pkl", "wb") as f:
            pickle.dump(snaps, f)
    table = pd.DataFrame(rows)
    table.to_csv(d / "partitions.csv", index=False)
    counts = table.groupby(["partition", "family"]).size().unstack()
    print(counts)
    assert table["episode_id"].is_unique

    man_path = d / "dataset_manifest.json"
    man = json.loads(man_path.read_text())
    ds = man["dataset"]
    ds["partitions"] = {"rule": "train->train; val->dev_cal/dev_thr; test->test_iid/cert; alternate within family in slot order",
                        "sizes": {k: len(v) for k, v in parts.items()}, "family_counts": counts.to_dict()}
    ds["prereg"] = {"file": args.prereg, "sha256": sha256_file(REPO_ROOT / args.prereg)}
    ds["files_sha256"] = {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.suffix in (".pkl", ".ttl", ".xlsx", ".csv")}
    write_json(man_path, man)
    print(json.dumps(ds["partitions"]["sizes"]))


if __name__ == "__main__":
    main()
